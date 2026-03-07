"""
数据流管理器 - 连接TCP数据流与解码器

对应需求：
    任务准备期间开始获取数据5s窗口，任务执行期间每隔100ms提取一次数据，并进行解码
    解码器检测到运动意图，输出刺激器指令（数据提供）
    解码日志的记录，包括接收到数据时间、数据shape（时间戳和数据信息）
"""
import sys
import os
import time
import warnings
import threading
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.transmission.trans_manager.remoteManagerTTL import RemoteManagerTTL


class DataStreamer:
    """
    数据流管理器
    """

    def __init__(self, remote_manager, config):
        """
        初始化数据流管理器
        :param remote_manager: RemoteManagerTTL实例，用于TCP数据获取
        :param config: 配置字典，包含采样率、通道数等参数
        """

        self.rm = remote_manager
        self.fs = config.get('fs', 2000)
        self.num_ch = config.get('num_ch', 128)

        # 任务执行时长
        self.buffer_duration = 10.0
        self.buffer_size = int(self.buffer_duration * self.fs)

        # 解码窗口大小和解码间隔
        self.decode_window_size = config.get('decode_window_size', 0.5)
        self.decode_interval = config.get('decode_interval', 0.1)

        self.preparation_window_size = config.get('preparation_window_size', 5.0)  # 5秒准备期

        self.is_recording = False  # 是否正在记录数据
        self.start_time = 0.0  #实验开始时间
        self.last_decode_time = -float('inf')  # 上次解码时间，用于控制100ms间隔


        self.data_stats = {
            'total_samples_received': 0,  # 总接收样本数
            'total_decode_windows': 0,  # 总解码窗口数
            'last_data_timestamp': 0.0,  # 最后一次接收数据的时间戳
            'last_data_shape': (0, 0)  # 最后一次接收数据的shape
        }

        self._lock = threading.Lock()

        print(f"DataStreamer初始化完成:")
        print(f"  - 采样率: {self.fs} Hz")
        print(f"  - 通道数: {self.num_ch}")
        print(f"  - 解码窗口: {self.decode_window_size} s")
        print(f"  - 解码间隔: {self.decode_interval} s")
        print(f"  - 准备期窗口: {self.preparation_window_size} s")

    def start_recording(self):
        """
        任务准备期间开始获取数据
        :return:
        """
        try:
            with self._lock:
                if self.is_recording:
                    warnings.warn("数据记录已经在运行中")
                    return True
                print("\n=== 启动数据流记录 ===")

                # 初始化设备连接
                self.rm.initialize_device(mode=0)  # mode=0: 同时使用命令和数据通道
                # 开始采集数据
                self.rm.begin_collect()
                # 记录开始时间
                self.start_time = time.time()
                self.is_recording = True
                # 获取系统信息并打印
                info = self.rm.get_info()
                print(f"数据采集已启动:")
                print(f"  - 采样率: {info['fs']} Hz")
                print(f"  - 通道数: {info['num_ch']}")
                print(f"  - 缓冲区大小: {info['num_buffer_sample']} 样本")
                print(f"  - 启动时间: {self.start_time:.3f}")

                return True

        except Exception as e:
            print(f"启动数据记录失败: {e}")
            return False

    def get_preparation_window(self, duration):
        """
        获取准备期数据窗口
        :param duration: 窗口时长
        :return: 数据数组，shape=(n_channels, n_timepoints)
        """

        if not self.is_recording:
            warnings.warn("数据记录未启动，请先调用start_recording()")
            return None

        if duration is None:
            duration = self.preparation_window_size

        try:
            # 计算需要获取的数据点数
            num_points = int(duration * self.fs)

            print(f"\n获取准备期数据窗口:")
            print(f"  - 时长: {duration} 秒")
            print(f"  - 数据点数: {num_points}")
            print(f"  - 期望shape: ({self.num_ch}, {num_points})")

            # 从TCP缓冲区获取数据
            data = self.rm.get_data(num_points)

            if data is None or data.size == 0:
                warnings.warn("获取准备期数据失败：返回数据为空")
                return None

            actual_shape = data.shape
            print(f"  - 实际shape: {actual_shape}")

            # 更新统计信息
            current_time = time.time()
            self.data_stats['last_data_timestamp'] = current_time - self.start_time
            self.data_stats['last_data_shape'] = actual_shape
            self.data_stats['total_samples_received'] += data.size

            return data

        except Exception as e:
            print(f"获取准备期数据窗口失败: {e}")
            return None

    def get_decode_window(self, window_size):
        """
        获取用于解码的数据窗口
        :param window_size: 解码窗口大小
        :return:
        (data, timestamp): 元组
            - data: numpy.ndarray
            - timestamp: 相对于实验开始的时间戳（秒）
        """
        if not self.is_recording:
            warnings.warn("数据记录未启动，请先调用start_recording()")
            return None, 0.0

        if window_size is None:
            window_size = self.decode_window_size

        # 计算当前时间（相对于实验开始）
        current_time = time.time()
        elapsed = current_time - self.start_time

        # 检查是否距上次解码超过100ms
        time_since_last_decode = elapsed - self.last_decode_time

        if time_since_last_decode < self.decode_interval:
            return None, elapsed

        try:
            # 计算需要获取的数据点数
            num_points = int(window_size * self.fs)

            # 从TCP缓冲区获取最新数据
            data = self.rm.get_data(num_points)

            if data is None or data.size == 0:
                return None, elapsed

            # 更新最后一次解码时间
            self.last_decode_time = elapsed

            # 更新统计信息
            self.data_stats['total_decode_windows'] += 1
            self.data_stats['last_data_timestamp'] = elapsed
            self.data_stats['last_data_shape'] = data.shape
            self.data_stats['total_samples_received'] += data.size

            # 打印调试信息
            if self.data_stats['total_decode_windows'] % 10 == 0:  # 每10次打印一次
                print(f"解码窗口 #{self.data_stats['total_decode_windows']}: "
                      f"time={elapsed:.2f}s, shape={data.shape}")

            return data, elapsed

        except Exception as e:
            print(f"获取解码窗口失败: {e}")
            return None, elapsed

    def get_latest_data(self, num_points):
        """
        获取最新数据
        :param num_points: 要获取的数据点数
        :return: 数据数组
        """

        if not self.is_recording:
            warnings.warn("数据记录未启动，请先调用start_recording()")
            return None

        try:
            data = self.rm.get_data(num_points)
            return data
        except Exception as e:
            print(f"获取最新数据失败: {e}")
            return None

    def check_buffer_status(self):
        """
        检查缓冲区状态
        :return:
        """
        try:
            info = self.rm.get_info()
            buffer_samples = info.get('num_buffer_sample', 0)
            buffer_time = buffer_samples / self.fs if self.fs > 0 else 0
            buffer_usage = buffer_time / self.buffer_size if self.buffer_size > 0 else 0

            # 警告阈值：80%
            is_warning = buffer_usage > 0.8

            status = {
                'buffer_time': buffer_time,
                'buffer_samples': buffer_samples,
                'buffer_usage': buffer_usage,
                'is_warning': is_warning,
                'info': info
            }

            if is_warning:
                print(f"⚠️  警告: 缓冲区使用率 {buffer_usage*100:.1f}%")

            return status

        except Exception as e:
            print(f"检查缓冲区状态失败: {e}")
            return {
                'buffer_time': 0,
                'buffer_usage': 0,
                'is_warning': True,
                'info': {}
            }

    def get_data_stats(self):
        """
        获取数据统计信息
        :return:
        """
        stats = self.data_stats.copy()

        # 计算录制时长
        if self.is_recording and self.start_time > 0:
            stats['recording_duration'] = time.time() - self.start_time
        else:
            stats['recording_duration'] = 0

        return stats

    def reset_decode_timer(self):
        """
        重置计时器
        :return:
        """
        self.last_decode_time = -float('inf')
        print("解码计时器已重置")

    def stop_recording(self):
        try:
            with self._lock:
                if not self.is_recording:
                    warnings.warn("数据记录未在运行")
                    return True

                print("\n=== 停止数据流记录 ===")

                # 打印最终统计
                stats = self.get_data_stats()
                print(f"数据统计:")
                print(f"  - 总接收样本数: {stats['total_samples_received']}")
                print(f"  - 总解码窗口数: {stats['total_decode_windows']}")
                print(f"  - 录制时长: {stats['recording_duration']:.2f} 秒")

                # 停止采集
                self.rm.stop_collect()

                # 关闭连接
                self.rm.close_connection(stop_collect=False)

                self.is_recording = False

                print("数据记录已停止")
                return True

        except Exception as e:
            print(f"停止数据记录失败: {e}")
            return False

    def __del__(self):
        if self.is_recording:
            try:
                self.stop_recording()
            except:
                pass


if __name__ == '__main__':

    pass
