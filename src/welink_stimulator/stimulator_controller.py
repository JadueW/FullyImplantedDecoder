import struct
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field

import serial


@dataclass
class StimulationParams:
    channel: str = 'A'
    current_ma: float = 1.0
    pulse_width: int = 2
    duration_min: int = 10

    def to_dict(self):
        return asdict(self)


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
    CMD_RESET = 0x34

    CHANNEL_MAP = {
        'A': 0x00,
        'B': 0x01,
        'AB': 0xFF,
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

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._stim_duration_thread = None
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

    def _send_command(self, cmd, data=b'', expect_response=True, response_timeout=None):
        if not self.is_connected and not self.connect():
            return None

        frame = self._build_frame(cmd, data)
        if self.debug:
            print(f'Stimulator request: {frame.hex(" ").upper()}')

        with self._lock:
            try:
                self.serial_conn.write(frame)
                self.serial_conn.flush()
                if not expect_response:
                    return b''
                return self._read_response(timeout=response_timeout)
            except Exception as exc:
                warnings.warn(f'Failed to send stimulator command: {exc}')
                return None

    def _normalize_params(self, params):
        if isinstance(params, StimulationParams):
            return params
        if isinstance(params, dict):
            return StimulationParams(**params)
        raise TypeError('params must be a StimulationParams or dict.')

    def set_stimulation_params(self, params):
        params = self._normalize_params(params)
        if params.channel not in self.CHANNEL_MAP:
            raise ValueError(f'Unsupported channel: {params.channel}')

        channel_byte = self.CHANNEL_MAP[params.channel]
        current_byte = max(0, min(255, int(round(params.current_ma))))
        pulse_byte = max(0, min(255, int(params.pulse_width)))
        duration_byte = max(0, min(255, int(params.duration_min)))

        if params.channel == 'AB':
            data = struct.pack(
                '<7B',
                channel_byte,
                current_byte,
                pulse_byte,
                duration_byte,
                current_byte,
                pulse_byte,
                duration_byte,
            )
        else:
            data = struct.pack(
                '<4B',
                channel_byte,
                current_byte,
                pulse_byte,
                duration_byte,
            )

        response = self._send_command(self.CMD_SET_PARAMS, data, expect_response=True)
        if response:
            self.current_params = params
            return True
        return False

    def _start_duration_timer(self, duration_ms):
        self._stop_event.clear()

        def timer_loop():
            if self._stop_event.wait(duration_ms / 1000.0):
                return
            if self.is_stimulating:
                self.stop_stimulation()

        self._stim_duration_thread = threading.Thread(
            target=timer_loop,
            name='StimDurationThread',
            daemon=True,
        )
        self._stim_duration_thread.start()

    def start_stimulation(self, duration_ms=None, channel=None):
        channel_code = self.CHANNEL_MAP.get(channel or getattr(self.current_params, 'channel', 'AB'), 0xFF)
        data = bytes([0x01, channel_code])
        response = self._send_command(self.CMD_START_STOP, data, expect_response=True)
        if not response:
            return False

        self.is_stimulating = True
        self._stim_start_time = time.time()
        if duration_ms is not None:
            self._start_duration_timer(duration_ms)
        return True

    def stop_stimulation(self, channel=None):
        self._stop_event.set()
        channel_code = self.CHANNEL_MAP.get(channel or getattr(self.current_params, 'channel', 'AB'), 0xFF)
        data = bytes([0x00, channel_code])
        response = self._send_command(self.CMD_START_STOP, data, expect_response=True)
        if not response:
            return False

        self.is_stimulating = False
        return True

    def reset_stimulation(self):
        response = self._send_command(self.CMD_RESET, b'', expect_response=True)
        if response:
            self.is_stimulating = False
            self.current_params = None
            return True
        return False

    def switch_channel(self, channel):
        if channel not in self.CHANNEL_MAP:
            raise ValueError(f'Unsupported channel: {channel}')
        response = self._send_command(
            self.CMD_SWITCH_CHANNEL,
            bytes([self.CHANNEL_MAP[channel]]),
            expect_response=True,
        )
        return bool(response)

    def send_command_with_duration(self, params, duration_ms):
        params = self._normalize_params(params)
        if not self.set_stimulation_params(params):
            return False
        return self.start_stimulation(duration_ms=duration_ms, channel=params.channel)

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
        if command.command_type == 'reset':
            return self.reset_stimulation()

        raise ValueError(f'Unsupported command_type: {command.command_type}')

    def query_status(self):
        response = self._send_command(self.CMD_STATUS, b'', expect_response=True)
        if response and len(response) >= 7:
            status_byte = response[5]
            return {
                'is_stimulating': bool(status_byte & 0x01),
                'raw_status': status_byte,
                'timestamp': time.time(),
            }
        return None

    def query_version(self):
        response = self._send_command(self.CMD_VERSION, b'', expect_response=True)
        if response and len(response) >= 11:
            fw_ver = response[6:9]
            hw_ver = response[9:11]
            return f'v{fw_ver[0]}.{fw_ver[1]}.{fw_ver[2]} (HW: v{hw_ver[0]}.{hw_ver[1]})'
        return None

    def power_on(self):
        return bool(self._send_command(self.CMD_POWER, bytes([0x01]), expect_response=True))

    def power_off(self):
        return bool(self._send_command(self.CMD_POWER, bytes([0x00]), expect_response=True))

    def get_info(self):
        return {
            'port': self.port,
            'baudrate': self.baudrate,
            'timeout': self.timeout,
            'is_connected': self.is_connected,
            'is_stimulating': self.is_stimulating,
            'current_params': self.current_params.to_dict() if self.current_params else None,
            'stimulation_duration_s': time.time() - self._stim_start_time if self.is_stimulating else 0.0,
        }

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
