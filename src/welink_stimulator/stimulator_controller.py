import threading
import time
import warnings
from dataclasses import asdict, dataclass, field

import serial


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


@dataclass
class ChannelStimConfig:
    route: int = 0
    prescription_id: int = 1
    level: int = 1
    duration_code: int = 1

    def normalized(self):
        return ChannelStimConfig(
            route=_clamp(self.route, 0, 7),
            prescription_id=_clamp(self.prescription_id, 1, 13),
            level=_clamp(self.level, 1, 32),
            duration_code=_clamp(self.duration_code, 1, 3),
        )

    def to_dict(self):
        return asdict(self.normalized())


def _build_channel_config(params, prefix):
    prefix_key = f'{prefix}_'
    return ChannelStimConfig(
        route=params.get(f'{prefix_key}route', params.get('route', 0)),
        prescription_id=params.get(
            f'{prefix_key}prescription_id',
            params.get('prescription_id', params.get('pulse_width', 1)),
        ),
        level=params.get(
            f'{prefix_key}level',
            params.get('level', params.get('current_ma', 1)),
        ),
        duration_code=params.get(
            f'{prefix_key}duration_code',
            params.get('duration_code', params.get('duration_min', 1)),
        ),
    )


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
        a_cfg = ChannelStimConfig(**params['a']) if isinstance(params.get('a'), dict) else _build_channel_config(params, 'a')
        b_cfg = ChannelStimConfig(**params['b']) if isinstance(params.get('b'), dict) else _build_channel_config(params, 'b')
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

    def _inactive_channel_config(self):
        return ChannelStimConfig(route=0, prescription_id=1, level=1, duration_code=1)

    def to_payload(self):
        normalized = self.normalized()
        a_cfg = normalized.a
        b_cfg = normalized.b

        if normalized.channel == 'A':
            b_cfg = self._inactive_channel_config()
        elif normalized.channel == 'B':
            a_cfg = self._inactive_channel_config()

        return bytes([
            StimulatorController.CHANNEL_MAP[normalized.channel],
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

    def __init__(self, port, baudrate=115200, timeout=0.05, debug=True):
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
        self._last_params_payload = None
        self._stats = {
            'total_commands': 0,
            'total_read_retries': 0,
            'total_discarded_bytes': 0,
            'total_time_ms': 0.0,
            'max_command_time_ms': 0.0,
        }

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
        return response[-2] == self._calculate_checksum(response[:-2])

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
            waiting = int(getattr(self.serial_conn, 'in_waiting', 0))
        except Exception:
            waiting = 0

        if waiting <= 0:
            return

        try:
            stale_bytes = self.serial_conn.read(waiting)
            self._stats['total_discarded_bytes'] += len(stale_bytes)
            if self.debug and stale_bytes:
                print(f'Discarded stale stimulator bytes: {stale_bytes.hex(" ").upper()}')
        except Exception as exc:
            warnings.warn(f'Failed to discard stale stimulator input: {exc}')

    def _read_response(self, timeout=None):
        if not self.serial_conn or not self.serial_conn.is_open:
            return None

        old_timeout = self.serial_conn.timeout
        if timeout is not None:
            self.serial_conn.timeout = timeout

        retry_count = 0
        try:
            while True:
                retry_count += 1
                head = self.serial_conn.read(1)
                if not head:
                    return None
                if head[0] == self.FRAME_HEAD:
                    break

            header = self.serial_conn.read(4)
            if len(header) != 4:
                return None

            _, _, len_h, len_l = header
            data_len = (len_h << 8) | len_l
            tail = self.serial_conn.read(data_len + 2)
            if len(tail) != data_len + 2:
                return None

            response = bytes([self.FRAME_HEAD]) + header + tail
            if not self._validate_response(response):
                warnings.warn('Stimulator response checksum/frame validation failed.')
                return None

            self._stats['total_read_retries'] += retry_count
            if self.debug:
                print(f'Stimulator response: {response.hex(" ").upper()}')
            return response
        finally:
            self.serial_conn.timeout = old_timeout

    def _send_command(self, cmd, data=b'', expected_response_cmd=None, response_timeout=None):
        if not self.is_connected and not self.connect():
            return None

        perf_start = time.perf_counter()
        frame = self._build_frame(cmd, data)

        with self._lock:
            try:
                self._discard_stale_input()
                if self.debug:
                    print(f'Stimulator request: {frame.hex(" ").upper()}')

                self.serial_conn.write(frame)
                self.serial_conn.flush()
                response = self._read_response(timeout=response_timeout)
                total_time = (time.perf_counter() - perf_start) * 1000.0

                self._stats['total_commands'] += 1
                self._stats['total_time_ms'] += total_time
                self._stats['max_command_time_ms'] = max(self._stats['max_command_time_ms'], total_time)

                if total_time > 200:
                    print(f'[Stimulator-SLOW] CMD=0x{cmd:02X} | Total:{total_time:.1f}ms')

                if response is None:
                    return None

                if expected_response_cmd is not None and response[2] != expected_response_cmd:
                    warnings.warn(
                        f'Unexpected response command: expected 0x{expected_response_cmd:02X}, '
                        f'got 0x{response[2]:02X}.'
                    )
                    return None
                return response
            except Exception as exc:
                total_time = (time.perf_counter() - perf_start) * 1000.0
                print(f'[Stimulator-ERROR] CMD=0x{cmd:02X} | Error after {total_time:.1f}ms | {exc}')
                warnings.warn(f'Failed to send stimulator command: {exc}')
                return None

    def _send_echo_command(self, cmd, payload, response_timeout=None):
        response = self._send_command(
            cmd,
            payload,
            expected_response_cmd=cmd,
            response_timeout=response_timeout,
        )
        response_payload = self._extract_payload(response, expected_cmd=cmd)
        return response_payload == bytes(payload)

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

    def _resolve_channel_name(self, channel=None):
        if channel is not None:
            channel_name = str(channel).upper()
        elif self.current_params is not None:
            channel_name = self.current_params.channel
        else:
            channel_name = 'AB'

        if channel_name not in self.CHANNEL_MAP:
            raise ValueError(f'Unsupported channel: {channel_name}')
        return channel_name

    def set_stimulation_params(self, params):
        params = self._normalize_params(params)
        payload = params.to_payload()

        if self._last_params_payload == payload:
            self.current_params = params
            return True

        if self._send_echo_command(self.CMD_SET_PARAMS, payload):
            self._last_params_payload = payload
            self.current_params = params
            return True
        return False

    def start_stimulation(self, params=None, duration_ms=None, channel=None):
        if params is not None and not self.set_stimulation_params(params):
            return False

        if self.current_params is None:
            warnings.warn('start_stimulation() requires valid stimulation params. Send 0x31 first.')
            return False

        channel_name = self._resolve_channel_name(channel)
        payload = bytes([0x01, self.CHANNEL_MAP[channel_name]])
        if not self._send_echo_command(self.CMD_START_STOP, payload):
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
        channel_name = self._resolve_channel_name(channel)
        payload = bytes([0x00, self.CHANNEL_MAP[channel_name]])
        if not self._send_echo_command(self.CMD_START_STOP, payload):
            return False

        self.is_stimulating = False
        return True

    def switch_channel(self, channel, a_route=0, b_route=0):
        channel_name = self._resolve_channel_name(channel)
        payload = bytes([
            self.CHANNEL_MAP[channel_name],
            _clamp(a_route, 0, 7),
            _clamp(b_route, 0, 7),
        ])
        return self._send_echo_command(self.CMD_SWITCH_CHANNEL, payload)

    def set_level_limit(self, enabled):
        payload = bytes([0x01 if bool(enabled) else 0x00])
        if self._send_echo_command(self.CMD_LEVEL_LIMIT, payload):
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
        return self.start_stimulation(params=params, duration_ms=duration_ms)

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
            return self.start_stimulation(params=command.params, duration_ms=command.duration_ms)
        if command.command_type == 'stop':
            return self.stop_stimulation(channel=command.params.channel)
        raise ValueError(f'Unsupported command_type: {command.command_type}')

    def query_status(self):
        response = self._send_command(self.CMD_STATUS, b'', expected_response_cmd=self.CMD_STATUS)
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
        response = self._send_command(self.CMD_VERSION, b'', expected_response_cmd=self.CMD_VERSION)
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
        return self._send_echo_command(self.CMD_POWER, bytes([0x00]))

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
            'stats': self.get_stats(),
        }

    def get_stats(self):
        if self._stats['total_commands'] > 0:
            avg_time = self._stats['total_time_ms'] / self._stats['total_commands']
            avg_retries = self._stats['total_read_retries'] / self._stats['total_commands']
        else:
            avg_time = 0.0
            avg_retries = 0.0

        return {
            'total_commands': self._stats['total_commands'],
            'avg_command_ms': round(avg_time, 2),
            'max_command_ms': round(self._stats['max_command_time_ms'], 2),
            'total_read_retries': self._stats['total_read_retries'],
            'avg_read_retries': round(avg_retries, 2),
            'total_discarded_bytes': self._stats['total_discarded_bytes'],
        }

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
