"""
通信协议：
- 串口通信（UART）
- 帧格式：EB + ADDR + CMD + LEN + DATA + CHECKSUM + 90
- 设备地址：0x01
主要指令：
- 0x20: 上电/下电系统电源
- 0x21: 查询设备版本信息
- 0x30: 查询采集状态
- 0x31: 设置采集刺激参数
- 0x32: 开始/停止采集刺激
- 0x33: 切换通道
- 0x34: 刺激复位
"""

import serial
import time
import struct
import threading
from dataclasses import dataclass
import warnings


@dataclass
class StimulationParams:
    """ 刺激参数 """
    channel = 'A'              # 通道: 'A', 'B', 或 'AB'
    current_ma = 1.0         # 电流强度 (mA)
    pulse_width = 2            # 脉宽
    duration_min = 10         # 刺激时长

    @classmethod
    def to_dict(cls):
        return {
            'channel': cls.channel,
            'current_ma': cls.current_ma,
            'pulse_width': cls.pulse_width,
            'duration_min': cls.duration_min
        }


@dataclass
class StimulusCommand:
    """ 刺激指令 """
    command_type = ""               # 指令类型: 'start', 'stop', 'set_params', 'reset'
    params = StimulationParams.to_dict()
    duration_ms = 500  # 持续时间（毫秒）


class StimulatorController:

    # 协议常量
    FRAME_HEAD = 0xEB               # 帧头
    FRAME_TAIL = 0x90               # 帧尾
    DEVICE_ADDR = 0x01              # 设备地址

    # 指令码
    CMD_POWER = 0x20                # 上电0x01/下电0x00
    CMD_VERSION = 0x21              # 查询版本
    CMD_STATUS = 0x30               # 查询状态
    CMD_SET_PARAMS = 0x31           # 设置参数
    CMD_START_STOP = 0x32           # 开始0x01/停止0x00
    CMD_SWITCH_CHANNEL = 0x33       # 切换通道
    CMD_RESET = 0x34                # 复位

    def __init__(self,port,baudrate,timeout,debug):
        """
        初始化刺激器控制器
        :param port:  串口号
        :param baudrate:  波特率
        :param timeout:  读超时
        :param debug:  是否打印调试信息
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug = debug

        # 串口对象
        self.serial_conn = None

        # 状态
        self.is_connected = False
        self.is_stimulating = False
        self.current_params = None

        # 线程锁
        self._lock = threading.Lock()

        # 计时器
        self._stim_start_time = 0.0
        self._stim_duration_thread = None
        self._stop_event = threading.Event()

        print(f"刺激器控制器初始化: {port} @ {baudrate} baud")

    def connect(self) :
        try:
            with self._lock:
                if self.is_connected:
                    warnings.warn("已经连接到刺激器")
                    return True

                if self.debug:
                    print(f"连接到刺激器: {self.port}")

                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )

                # 清空缓冲区
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()

                self.is_connected = True

                if self.debug:
                    print("刺激器连接成功")

                return True

        except Exception as e:
            print(f"连接刺激器失败: {e}")
            return False

    def disconnect(self):
        try:
            with self._lock:
                if self.is_stimulating:
                    self.stop_stimulation()

                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.close()

                self.is_connected = False

                if self.debug:
                    print("刺激器已断开")

                return True

        except Exception as e:
            print(f"断开刺激器失败: {e}")
            return False

    def _calculate_checksum(self, data):
        """
        计算checksum
        :param data: 数据字节
        :return: 校验和
        """
        total = sum(data)
        checksum = ((~total) + 1) & 0xFF
        return checksum

    def _build_frame(self,cmd,data) :
        """
        构建通信帧
        :param cmd:指令码
        :param data: 具体数据内容
        :return: 完整的数据帧结构
        """
        # 帧结构: EB + ADDR + CMD + LEN(2) + DATA + CHECKSUM + 90
        data_len = len(data)

        # 构建帧
        frame_without_checksum = bytes([
            self.FRAME_HEAD,
            self.DEVICE_ADDR,
            cmd,
            (data_len >> 8) & 0xFF,  # 长度高字节
            data_len & 0xFF           # 长度低字节
        ]) + data

        # 计算校验和
        checksum = self._calculate_checksum(frame_without_checksum)

        # 添加校验和和帧尾
        frame = frame_without_checksum + bytes([checksum, self.FRAME_TAIL])

        return frame

    def _send_command(self,cmd,data,expect_response):
        """
        发送命令
        :param cmd:指令码
        :param data:数据
        :param expect_response:是否期望响应
        :return:如果expect_response=True的响应数据
        """
        if not self.is_connected:
            warnings.warn("未连接到刺激器")
            return None

        try:
            # 构建帧
            frame = self._build_frame(cmd, data)

            if self.debug:
                print(f"发送命令: {frame.hex(' ').upper()}")

            # 发送
            with self._lock:
                self.serial_conn.write(frame)
                self.serial_conn.flush()

            # 接收响应
            if expect_response:
                response = self._read_response()
                return response

            return None

        except Exception as e:
            print(f"发送命令失败: {e}")
            return None

    def _read_response(self, timeout):
        try:
            if timeout is not None:
                old_timeout = self.serial_conn.timeout
                self.serial_conn.timeout = timeout

            # 等待帧头
            while True:
                byte = self.serial_conn.read(1)
                if not byte:
                    if self.debug:
                        print("读取响应超时")
                    return None
                if byte[0] == self.FRAME_HEAD:
                    break

            # 读取剩余部分
            # ADDR + CMD + LEN_H + LEN_L
            header = self.serial_conn.read(4)
            if len(header) < 4:
                return None

            addr, cmd, len_h, len_l = header
            data_len = (len_h << 8) | len_l

            # 读取数据 + 校验和 + 帧尾
            tail = self.serial_conn.read(data_len + 2)
            if len(tail) < data_len + 2:
                return None

            # 组合完整响应
            response = bytes([self.FRAME_HEAD]) + header + tail

            if self.debug:
                print(f"接收响应: {response.hex(' ').upper()}")

            # 验证帧尾
            if response[-1] != self.FRAME_TAIL:
                warnings.warn(f"无效的帧尾: 0x{response[-1]:02X}")
                return None

            # 验证校验和
            # TODO: 可以添加校验和验证

            return response

        except Exception as e:
            print(f"读取响应失败: {e}")
            return None
        finally:
            if timeout is not None and self.serial_conn:
                self.serial_conn.timeout = old_timeout

    def set_stimulation_params(self,params):
        """
        设置刺激参数
        :param params:刺激参数
        :return: 是否设置成功
        Byte 0: 通道选择
            - 0xFF: A/B同时
            - 0x00: A通道
            - 0x01: B通道
        Byte 1: 电流强度 (mA)
        Byte 2: 脉宽
        Byte 3: 刺激时长 (分钟)
        Bytes 4-7: B通道参数（如果同时使用）
        """

        try:
            # 通道编码
            channel_map = {
                'A': 0x00,
                'B': 0x01,
                'AB': 0xFF
            }

            if params.channel not in channel_map:
                raise ValueError(f"无效的通道: {params.channel}")

            channel_byte = channel_map[params.channel]

            # 构建数据
            if params.channel == 'AB':
                # A/B通道参数
                data = struct.pack(
                    '<B8B',  # 小端序
                    channel_byte,              # 通道
                    int(params.current_ma),    # A电流
                    params.pulse_width,        # A脉宽
                    params.duration_min,       # A时长
                    int(params.current_ma),    # B电流
                    params.pulse_width,        # B脉宽
                    params.duration_min        # B时长
                )
            else:
                # 单通道参数（4字节）
                data = struct.pack(
                    '<BBBB',
                    channel_byte,
                    int(params.current_ma),
                    params.pulse_width,
                    params.duration_min
                )

            # 发送命令
            response = self._send_command(self.CMD_SET_PARAMS, data,True)

            if response:
                self.current_params = params
                if self.debug:
                    print(f"刺激参数已设置: {params.to_dict()}")
                return True

            return False

        except Exception as e:
            print(f"设置刺激参数失败: {e}")
            return False

    def start_stimulation(self,duration_ms) :
        """
        开始刺激
        :param duration_ms: 刺激持续时间（毫秒）
        :return:
        """
        try:
            # 数据格式：Byte 0 = 0x01(开始), Byte 1 = 通道掩码
            data = bytes([0x01, 0xFF])  # 开始，A/B通道

            response = self._send_command(self.CMD_START_STOP, data,True)

            if response:
                self.is_stimulating = True
                self._stim_start_time = time.time()

                if self.debug:
                    print(f"刺激已开始")

                if duration_ms is not None:
                    self._start_duration_timer(duration_ms)

                return True

            return False

        except Exception as e:
            print(f"开始刺激失败: {e}")
            return False

    def _start_duration_timer(self, duration_ms: float):
        """启动持续时间计时器"""
        self._stop_event.clear()

        def timer_thread():
            wait_time = duration_ms / 1000.0
            if self._stop_event.wait(timeout=wait_time):
                return  # 被提前停止

            # 时间到，自动停止刺激
            if self.is_stimulating:
                self.stop_stimulation()
                if self.debug:
                    print(f"刺激持续时间({duration_ms}ms)已到，自动停止")

        self._stim_duration_thread = threading.Thread(
            target=timer_thread,
            daemon=True
        )
        self._stim_duration_thread.start()

    def stop_stimulation(self):
        try:
            # 停止计时器线程
            self._stop_event.set()

            # 数据格式：Byte 0 = 0x00(停止), Byte 1 = 通道掩码
            data = bytes([0x00, 0xFF])  # 停止，A/B通道

            response = self._send_command(self.CMD_START_STOP, data,True)

            if response:
                self.is_stimulating = False

                if self.debug:
                    print("刺激已停止")

                return True

            return False

        except Exception as e:
            print(f"停止刺激失败: {e}")
            return False

    def reset_stimulation(self):
        try:
            response = self._send_command(self.CMD_RESET,data=[],expect_response=True)

            if response:
                self.is_stimulating = False
                self.current_params = None

                if self.debug:
                    print("刺激器已复位")

                return True

            return False

        except Exception as e:
            print(f"复位刺激器失败: {e}")
            return False

    def send_command_with_duration(self,params,duration_ms) :

        try:
            # 1. 设置参数
            if not self.set_stimulation_params(params):
                return False

            # 2. 开始刺激（会自动在duration_ms后停止）
            if not self.start_stimulation(duration_ms=duration_ms):
                return False

            return True

        except Exception as e:
            print(f"发送刺激指令失败: {e}")
            return False

    def query_status(self):
        """
        查询设备状态
        :return:
        """
        try:
            response = self._send_command(self.CMD_STATUS,[],True)

            if response and len(response) >= 6:
                status_byte = response[5]

                status = {
                    'is_stimulating': bool(status_byte & 0x01),
                    'raw_status': status_byte,
                    'timestamp': time.time()
                }

                return status

            return None

        except Exception as e:
            print(f"查询状态失败: {e}")
            return None

    def query_version(self):
        """
        查询设备版本
        """
        try:
            response = self._send_command(self.CMD_VERSION,[],True)

            if response and len(response) >= 10:
                # 解析版本信息
                # 响应格式: EB + ADDR + CMD + LEN + PROTOCOL + FW_VER + HW_VER + ...
                protocol_ver = response[5]
                fw_ver = response[6:9]  # 3字节
                hw_ver = response[9:11]  # 2字节

                version = f"v{fw_ver[0]}.{fw_ver[1]}.{fw_ver[2]} (HW: v{hw_ver[0]}.{hw_ver[1]})"

                if self.debug:
                    print(f"设备版本: {version}")

                return version

            return None

        except Exception as e:
            print(f"查询版本失败: {e}")
            return None

    def power_on(self) :
        """
        上电
        """
        try:
            data = bytes([0x01])  # 上电
            response = self._send_command(self.CMD_POWER, data,True)

            if response:
                if self.debug:
                    print("设备已上电")
                return True

            return False

        except Exception as e:
            print(f"上电失败: {e}")
            return False

    def power_off(self) -> bool:
        """
        下电
        """
        try:
            data = bytes([0x00])  # 下电
            response = self._send_command(self.CMD_POWER, data,True)

            if response:
                if self.debug:
                    print("设备已下电")
                return True

            return False

        except Exception as e:
            print(f"下电失败: {e}")
            return False

    def get_info(self):
        """
        获取控制器信息
        """
        return {
            'port': self.port,
            'baudrate': self.baudrate,
            'is_connected': self.is_connected,
            'is_stimulating': self.is_stimulating,
            'current_params': self.current_params.to_dict() if self.current_params else None,
            'stimulation_duration_s': time.time() - self._stim_start_time if self.is_stimulating else 0.0
        }

    def __del__(self):
        """析构函数"""
        if self.is_connected:
            try:
                self.disconnect()
            except:
                pass


