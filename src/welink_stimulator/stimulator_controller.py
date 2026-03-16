import time
import warnings
from dataclasses import asdict, dataclass, field
import threading

import serial


@dataclass
class ChannelStimConfig:
    route: int = 0
    prescription_id: int = 1
    level: int = 1
    duration_code: int = 1

    def normalized(self):
        return ChannelStimConfig(
            route=max(0, min(7, int(self.route))),
            prescription_id=max(1, min(13, int(self.prescription_id))),
            level=max(1, min(32, int(self.level))),
            duration_code=max(1, min(3, int(self.duration_code))),
        )

    def to_dict(self):
        return asdict(self.normalized())


@dataclass
class StimulationParams:
    channel: str = 'A'
    a: ChannelStimConfig = field(default_factory=ChannelStimConfig)
    b: ChannelStimConfig = field(default_factory=ChannelStimConfig)

    @classmethod
    def from_mapping(cls, mapping):
        if isinstance(mapping, cls):
            return mapping

        params = dict(mapping)
        channel = str(params.get('channel', 'A')).upper()

        if 'a' in params and isinstance(params['a'], dict):
            a_cfg = ChannelStimConfig(**params['a'])
        else:
            a_cfg = ChannelStimConfig(
                route=params.get('a_route', params.get('route', 0)),
                prescription_id=params.get(
                    'a_prescription_id',
                    params.get(
                        'prescription_id',
                        params.get('pulse_width', 1),
                    ),
                ),
                level=params.get(
                    'a_level',
                    params.get('level', params.get('current_ma', 1)),
                ),
                duration_code=params.get(
                    'a_duration_code',
                    params.get('duration_code', params.get('duration_min', 1)),
                ),
            )

        if 'b' in params and isinstance(params['b'], dict):
            b_cfg = ChannelStimConfig(**params['b'])
        else:
            b_cfg = ChannelStimConfig(
                route=params.get('b_route', params.get('route', 0)),
                prescription_id=params.get(
                    'b_prescription_id',
                    params.get(
                        'prescription_id',
                        params.get('pulse_width', 1),
                    ),
                ),
                level=params.get(
                    'b_level',
                    params.get('level', params.get('current_ma', 1)),
                ),
                duration_code=params.get(
                    'b_duration_code',
                    params.get('duration_code', params.get('duration_min', 1)),
                ),
            )

        return cls(channel=channel, a=a_cfg, b=b_cfg)

    def normalized(self):
        channel = str(self.channel).upper()
        if channel not in {'A', 'B', 'AB'}:
            raise ValueError(f'Unsupported channel: {self.channel}')

        return StimulationParams(
            channel=channel,
            a=self.a.normalized(),
            b=self.b.normalized(),
        )

    def to_payload(self):
        normalized = self.normalized()
        selector = StimulatorController.CHANNEL_MAP[normalized.channel]

        a_cfg = normalized.a
        b_cfg = normalized.b

        if normalized.channel == 'A':
            b_cfg = ChannelStimConfig(route=0, prescription_id=1, level=1, duration_code=1)
        elif normalized.channel == 'B':
            a_cfg = ChannelStimConfig(route=0, prescription_id=1, level=1, duration_code=1)

        return bytes([
            selector,
            a_cfg.route,
            a_cfg.prescription_id,
            a_cfg.level,
            a_cfg.duration_code,
            b_cfg.route,
            b_cfg.prescription_id,
            b_cfg.level,
            b_cfg.duration_code,
        ])

    def to_dict(self):
        normalized = self.normalized()
        return {
            'channel': normalized.channel,
            'a': normalized.a.to_dict(),
            'b': normalized.b.to_dict(),
        }


@dataclass
class StimulusCommand:
    command_type: str = 'start'
    params: StimulationParams = field(default_factory=StimulationParams)
    duration_ms: int = 500

    def to_dict(self):
        payload = asdict(self)
        payload['params'] = self.params.to_dict()
        return payload


class StimulatorController:
    FRAME_HEAD = 0xEB
    FRAME_TAIL = 0x90
    DEVICE_ADDR = 0x01

    CMD_POWER = 0x20
    CMD_VERSION = 0x21
    CMD_STATUS = 0x30
    CMD_SET_PARAMS = 0x31
    CMD_START_STOP = 0x32
    CMD_SWITCH_CHANNEL = 0x33
    CMD_LEVEL_LIMIT = 0x34

    CHANNEL_MAP = {
        'A': 0x00,
        'B': 0x01,
        'AB': 0xFF,
    }

    DURATION_CODE_TO_MIN = {
        0x01: 10,
        0x02: 20,
        0x03: 30,
    }

    def __init__(self, port, baudrate=115200, timeout=1.0, debug=False):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug = debug

        self.serial_conn = None
        self.is_connected = False
        self.is_stimulating = False
        self.current_params = None
        self.level_limit_enabled = None

        self._lock = threading.RLock()
        self._stim_start_time = 0.0

    def connect(self):
        with self._lock:
            if self.is_connected and self.serial_conn and self.serial_conn.is_open:
                return True

            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                try:
                    self.serial_conn.setDTR(True)
                except Exception:
                    pass
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                self.is_connected = True
                return True
            except Exception as exc:
                warnings.warn(f'Failed to connect stimulator: {exc}')
                self.is_connected = False
                return False

    def disconnect(self):
        with self._lock:
            try:
                if self.is_stimulating:
                    self.stop_stimulation()

                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.close()

                self.serial_conn = None
                self.is_connected = False
                return True
            except Exception as exc:
                warnings.warn(f'Failed to disconnect stimulator: {exc}')
                return False

    def _calculate_checksum(self, data):
        return ((~sum(data)) + 1) & 0xFF

    def _build_frame(self, cmd, data=b''):
        payload = bytes(data)
        data_len = len(payload)
        frame_without_checksum = bytes([
            self.FRAME_HEAD,
            self.DEVICE_ADDR,
            cmd,
            (data_len >> 8) & 0xFF,
            data_len & 0xFF,
        ]) + payload
        checksum = self._calculate_checksum(frame_without_checksum)
        return frame_without_checksum + bytes([checksum, self.FRAME_TAIL])

    def _validate_response(self, response):
        if response is None or len(response) < 7:
            return False
        if response[0] != self.FRAME_HEAD or response[-1] != self.FRAME_TAIL:
            return False

        checksum = response[-2]
        expected_checksum = self._calculate_checksum(response[:-2])
        return checksum == expected_checksum

    def _read_response(self, timeout=None):
        if not self.serial_conn or not self.serial_conn.is_open:
            return None

        old_timeout = self.serial_conn.timeout
        if timeout is not None:
            self.serial_conn.timeout = timeout

        try:
            while True:
                head = self.serial_conn.read(1)
                if not head:
                    return None
                if head[0] == self.FRAME_HEAD:
                    break

            header = self.serial_conn.read(4)
            if len(header) < 4:
                return None

            _, _, len_h, len_l = header
            data_len = (len_h << 8) | len_l
            tail = self.serial_conn.read(data_len + 2)
            if len(tail) < data_len + 2:
                return None

            response = bytes([self.FRAME_HEAD]) + header + tail
            if self.debug:
                print(f'Stimulator response: {response.hex(" ").upper()}')

            if not self._validate_response(response):
                warnings.warn('Stimulator response checksum/frame validation failed.')
                return None
            return response
        finally:
            self.serial_conn.timeout = old_timeout

    def _extract_payload(self, response, expected_cmd=None):
        if not response:
            return None
        if expected_cmd is not None and response[2] != expected_cmd:
            warnings.warn(
                f'Unexpected response command: expected 0x{expected_cmd:02X}, '
                f'got 0x{response[2]:02X}.'
            )
            return None

        data_len = (response[3] << 8) | response[4]
        return response[5:5 + data_len]

    def _discard_stale_input(self):
        if not self.serial_conn or not self.serial_conn.is_open:
            return

        try:
            waiting = int(getattr(self.serial_conn, "in_waiting", 0))
        except Exception:
            waiting = 0

        if waiting <= 0:
            return

        try:
            stale_bytes = self.serial_conn.read(waiting)
            if self.debug and stale_bytes:
                print(f'Discarded stale stimulator bytes: {stale_bytes.hex(" ").upper()}')
        except Exception as exc:
            warnings.warn(f'Failed to discard stale stimulator input: {exc}')

    def _send_command(
        self,
        cmd,
        data=b'',
        expect_response=True,
        response_timeout=None,
        expected_response_cmd=None,
    ):
        if not self.is_connected and not self.connect():
            return None

        frame = self._build_frame(cmd, data)
        if self.debug:
            print(f'Stimulator request: {frame.hex(" ").upper()}')

        with self._lock:
            try:
                if expect_response:
                    self._discard_stale_input()
                self.serial_conn.write(frame)
                self.serial_conn.flush()
                if not expect_response:
                    return b''
                deadline = None
                if response_timeout is not None:
                    deadline = time.perf_counter() + float(response_timeout)

                while True:
                    remaining = None
                    if deadline is not None:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            return None

                    response = self._read_response(timeout=remaining)
                    if response is None:
                        return None

                    if expected_response_cmd is None or response[2] == expected_response_cmd:
                        return response

                    # start/stop commands are sent fire-and-forget, but some devices still
                    # emit a delayed 0x32 response that can arrive just before the next
                    # response-expected command (for example 0x31 set_params). Treat that
                    # specific pattern as benign stale input and drop it silently unless
                    # debug logging is enabled.
                    if (
                        expected_response_cmd == self.CMD_SET_PARAMS
                        and response[2] == self.CMD_START_STOP
                    ):
                        if self.debug:
                            print(
                                "Discarded stale 0x32 response before expected 0x31 response."
                            )
                        continue

                    warnings.warn(
                        f'Unexpected response command: expected 0x{expected_response_cmd:02X}, '
                        f'got 0x{response[2]:02X}; skipping stale response.'
                    )
            except Exception as exc:
                warnings.warn(f'Failed to send stimulator command: {exc}')
                return None

    def _normalize_params(self, params):
        if isinstance(params, StimulationParams):
            return params.normalized()
        if isinstance(params, dict):
            if any(key in params for key in ('current_ma', 'pulse_width', 'duration_min')):
                warnings.warn(
                    'Legacy stimulation parameter keys detected. '
                    'Please migrate to route/prescription_id/level/duration_code.'
                )
            return StimulationParams.from_mapping(params).normalized()
        raise TypeError('params must be a StimulationParams or dict.')

    def set_stimulation_params(self, params):
        params = self._normalize_params(params)

        response = self._send_command(
            self.CMD_SET_PARAMS,
            params.to_payload(),
            expect_response=True,
            expected_response_cmd=self.CMD_SET_PARAMS,
        )
        payload = self._extract_payload(response, expected_cmd=self.CMD_SET_PARAMS)
        if payload == params.to_payload():
            self.current_params = params
            return True
        return False

    def start_stimulation(self, duration_ms=None, channel=None):
        channel_name = str(channel or getattr(self.current_params, 'channel', 'AB')).upper()
        channel_code = self.CHANNEL_MAP.get(channel_name, 0xFF)
        response = self._send_command(
            self.CMD_START_STOP,
            bytes([0x01, channel_code]),
            expect_response=False,
        )
        if response is None:
            return False

        self.is_stimulating = True
        self._stim_start_time = time.time()
        if duration_ms is not None:
            warnings.warn(
                'start_stimulation(duration_ms=...) no longer auto-stops. '
                'Use explicit stop_stimulation() from the caller when the duration elapses.'
            )
        return True

    def stop_stimulation(self, channel=None):
        channel_name = str(channel or getattr(self.current_params, 'channel', 'AB')).upper()
        channel_code = self.CHANNEL_MAP.get(channel_name, 0xFF)
        expected_payload = bytes([0x00, channel_code])
        response = self._send_command(
            self.CMD_START_STOP,
            expected_payload,
            expect_response=False,
        )
        if response is None:
            return False

        self.is_stimulating = False
        return True

    def switch_channel(self, channel, a_route=0, b_route=0):
        channel_name = str(channel).upper()
        if channel_name not in self.CHANNEL_MAP:
            raise ValueError(f'Unsupported channel: {channel}')

        payload = bytes([
            self.CHANNEL_MAP[channel_name],
            max(0, min(7, int(a_route))),
            max(0, min(7, int(b_route))),
        ])
        response = self._send_command(
            self.CMD_SWITCH_CHANNEL,
            payload,
            expect_response=True,
            expected_response_cmd=self.CMD_SWITCH_CHANNEL,
        )
        response_payload = self._extract_payload(response, expected_cmd=self.CMD_SWITCH_CHANNEL)
        return response_payload == payload

    def set_level_limit(self, enabled):
        payload = bytes([0x01 if bool(enabled) else 0x00])
        response = self._send_command(
            self.CMD_LEVEL_LIMIT,
            payload,
            expect_response=True,
            expected_response_cmd=self.CMD_LEVEL_LIMIT,
        )
        response_payload = self._extract_payload(response, expected_cmd=self.CMD_LEVEL_LIMIT)
        if response_payload == payload:
            self.level_limit_enabled = bool(enabled)
            return True
        return False

    def reset_stimulation(self):
        warnings.warn(
            'Protocol 0x34 is level-limit control, not reset. '
            'Use set_level_limit(...) or stop_stimulation() instead.'
        )
        return False

    def send_command_with_duration(self, params, duration_ms):
        params = self._normalize_params(params)
        if not self.set_stimulation_params(params):
            return False
        warnings.warn(
            'send_command_with_duration() now only sends set_params + start. '
            'The caller must send stop_stimulation() after the desired duration.'
        )
        return self.start_stimulation(channel=params.channel)

    def execute_command(self, command):
        if isinstance(command, dict):
            command = StimulusCommand(
                command_type=command.get('command_type', 'start'),
                params=self._normalize_params(command.get('params', {})),
                duration_ms=int(command.get('duration_ms', 500)),
            )

        if not isinstance(command, StimulusCommand):
            raise TypeError('command must be a StimulusCommand or dict.')

        if command.command_type == 'set_params':
            return self.set_stimulation_params(command.params)
        if command.command_type == 'start':
            return self.send_command_with_duration(command.params, command.duration_ms)
        if command.command_type == 'stop':
            return self.stop_stimulation(channel=command.params.channel)

        raise ValueError(f'Unsupported command_type: {command.command_type}')

    def query_status(self):
        response = self._send_command(self.CMD_STATUS, b'', expect_response=True)
        payload = self._extract_payload(response, expected_cmd=self.CMD_STATUS)
        if payload is None or len(payload) != 18:
            return None

        hardware_status = (payload[1] << 8) | payload[2]
        battery_mv = (payload[3] << 8) | payload[4]

        return {
            'work_state': payload[0],
            'is_stimulating': payload[0] == 0x01,
            'hardware_status_raw': hardware_status,
            'is_charging': bool(hardware_status & 0x0001),
            'is_low_battery': bool(hardware_status & 0x0002),
            'battery_mv': battery_mv,
            'battery_v': battery_mv / 1000.0,
            'battery_soc': payload[5],
            'a_channel': {
                'is_stimulating': bool(payload[6]),
                'electrode_detached': bool(payload[7]),
                'route': payload[8],
                'prescription_id': payload[9],
                'level': payload[10],
                'remaining_min': payload[11],
            },
            'b_channel': {
                'is_stimulating': bool(payload[12]),
                'electrode_detached': bool(payload[13]),
                'route': payload[14],
                'prescription_id': payload[15],
                'level': payload[16],
                'remaining_min': payload[17],
            },
            'timestamp': time.time(),
        }

    def query_version(self):
        response = self._send_command(self.CMD_VERSION, b'', expect_response=True)
        payload = self._extract_payload(response, expected_cmd=self.CMD_VERSION)
        if payload is None or len(payload) != 7:
            return None

        protocol_raw = (payload[0] << 8) | payload[1]
        protocol_major = protocol_raw // 100
        protocol_minor = (protocol_raw % 100) // 10
        protocol_patch = protocol_raw % 10

        return {
            'protocol_raw': protocol_raw,
            'protocol_version': f'v{protocol_major}.{protocol_minor}.{protocol_patch}',
            'firmware_version': f'v{payload[2]}.{payload[3]}.{payload[4]}',
            'hardware_version': f'v{payload[5]}.{payload[6]}',
        }

    def power_on(self):
        warnings.warn(
            'The protocol documentation only defines the 0x20 shutdown command. '
            'power_on() is not supported over serial.'
        )
        return False

    def power_off(self):
        payload = bytes([0x00])
        response = self._send_command(self.CMD_POWER, payload, expect_response=True)
        response_payload = self._extract_payload(response, expected_cmd=self.CMD_POWER)
        return response_payload == payload

    def get_info(self):
        return {
            'port': self.port,
            'baudrate': self.baudrate,
            'timeout': self.timeout,
            'is_connected': self.is_connected,
            'is_stimulating': self.is_stimulating,
            'current_params': self.current_params.to_dict() if self.current_params else None,
            'level_limit_enabled': self.level_limit_enabled,
            'stimulation_duration_s': time.time() - self._stim_start_time if self.is_stimulating else 0.0,
        }

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
