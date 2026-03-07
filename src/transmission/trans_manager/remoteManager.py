# -*- coding:utf-8 -*-
# author:70706
# datetime:2024/11/27 17:09
# software: PyCharm
import sys
import os
import time
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from transmission.trans_control.commandController import CmdController
from transmission.trans_control.dataController import DataController


class RemoteManager:
    def __init__(self, host):
        self.host = host
        self.cmd_server_port = 5000
        self.data_server_port = 5001

        self.cmd_controller: CmdController = None
        self.data_controller: DataController = None

        # 启动时间
        self.startup_time = 0.

        self.fs = 0.
        self.num_ch = 0
        self.num_stream = 0
        self.duration = 0.
        self.num_buffer_sample = 0
        self.num_marker = 0
        self.__info = {
            "fs": self.fs,
            "num_ch": self.num_ch,
            "duration": self.duration,
            "num_buffer_sample": self.num_buffer_sample,
            "num_marker": self.num_marker,
        }

    def initialize_device(self, mode=0):
        """
        建立连接
        """
        if mode == 0:
            is_use_cmd = True
            is_use_data = True
        elif mode == 1:
            is_use_cmd = True
            is_use_data = False
        elif mode == 2:
            is_use_cmd = False
            is_use_data = True
        else:
            raise ValueError("Invalid mode. mode should be 0, 1, 2.")
        if is_use_cmd:
            # 首先建立cmd server连接
            self.cmd_controller = CmdController(self.host, self.cmd_server_port)
            self.cmd_controller.connect()
            self.cmd_controller.set_blocking(True)
            self.cmd_controller.set_timeout(0.1)
            # 拿到配置信息，包括fs, num_stream=int(num_ch/32),并更新self.info
            self.get_info()
        if is_use_data:
            # 与data server连接
            self.data_controller = DataController(self.host, int(self.data_server_port), int(self.num_stream),
                                                  int(self.fs))
            print(self.host, self.data_server_port, self.num_stream, self.fs)
            self.data_controller.connect()
            # 设置缓冲区的大小为10s
            self.data_controller.setBufferSize(10)

        print("Initialized...")

    def set_notion(self, notion: str = ""):
        """
        设置说明信息, 无固定结构
        """
        # 设置notion长度，不能超过1024个字符
        if len(notion) <= 1024:
            self.cmd_controller.set_notion(notion)
        else:
            warnings.warn(
                "Notion length exceeded the limit of 1024 characters."
            )

    def begin_collect(self):
        """
        开始采集
        """
        if self.cmd_controller:
            # 首先判断采集软件是否为停止状态
            _status = self.get_status()
            if _status.lower() != "stop":
                # 非停止，首先需要暂停停止运行
                self.stop_collect()

        if self.data_controller:
            # 先让data controller线程启动，准备持续接收
            self.data_controller.start()
            time.sleep(0.1)
            print("Data receiving thread starting...")

        if self.cmd_controller:
            # 启动软件
            self.cmd_controller.start()
            print("Start Recording...")

        self.startup_time = time.time()

    def stop_collect(self):
        """
        停止采集
        不清空缓冲
        """
        if self.cmd_controller:
            self.cmd_controller.stop()
            self.cmd_controller = None
        print("Stop Recording...")

    def get_data(self, num_point: int = -1):
        """
        获取数据
        从data_controller中提取数据
        """
        if self.data_controller:
            if num_point <= 0:
                num_point = -1

            # 如果参数大于缓冲区的点数，只能打警告，并返回缓冲区所有数据
            _buffer_len = int(self.data_controller.getBufferTime() * self.fs)
            # print(f"Buffer len: {_buffer_len}")
            if num_point > _buffer_len:
                warnings.warn(
                    f"Expected length {num_point} exceeded the length of the buffer {_buffer_len}."
                )
                num_point = _buffer_len

            _data = self.data_controller.get_data(int(num_point))
            return _data
        else:
            return np.array([])

    def set_marker(self, marker_id: int = 1):
        """
        设置marker
        入参：id,
        最后 marker的结构List[[id, ref_time, timestamp]]
        timestamp= fs * ref_time
        """
        if self.cmd_controller is None:
            print("Please check if the cmd tcp connected.")
            return
        if isinstance(marker_id, int):
            if marker_id < 0:
                marker_id = 0
            elif marker_id >= 8:
                marker_id = 8
        else:
            warnings.warn(f"Expected marker_id of int, got type(marker_id) instead. marker_id was set to 0.")
            marker_id = 0
        # 将marker_id标记为字节位为高
        marker_bytes_map = [2 ** i for i in range(8)]
        marker_bytes_map.insert(0, 0)

        marker_bytes_value = marker_bytes_map[marker_id]
        return self.cmd_controller.set_marker(marker_bytes_value)

    def get_marker(self):
        """
        获取marker列表
        从服务器拿到所有marker
        最后 marker的结构List[[id, ref_time, timestamp]]
        timestamp= fs * ref_time
        """
        if self.cmd_controller is None:
            print("Please check if the cmd tcp connected.")
            return []
        _marker_list = self.cmd_controller.get_marker()
        return _marker_list

    def get_info(self):
        """
        获取信息，包括缓冲区现在长度、marker个数、数据采集时长、
        需要调用get_marker()
        """
        if self.cmd_controller:
            if self.fs <= 0:
                self.fs = self.cmd_controller.get_fs()
            if not self.num_stream:
                num_channel = self.cmd_controller.get_num_channel()
                # ch_info = (PORT, NUM_CH_PER_PORT)
                self.num_ch = num_channel
                self.num_stream = int(self.num_ch / 32)

            self.duration = self.cmd_controller.get_duration()

            # self.num_marker = self.cmd_controller.get_num_marker()
            self.num_marker = len(self.get_marker())
        if self.data_controller:
            try:
                self.num_buffer_sample = self.data_controller.get_buffer_size()
            except AttributeError:
                self.num_buffer_sample = 0

        self.__info.update(
            fs=self.fs,
            num_ch=self.num_ch,
            duration=self.duration,
            num_buffer_sample=self.num_buffer_sample,
            num_marker=self.num_marker,
        )
        return self.__info

    def get_status(self):
        """
        获取系统当前运行状态
        发送命令
        """
        if self.cmd_controller is None:
            print("Please check if the cmd tcp connected.")
            return ""
        _cur_status = self.cmd_controller.get_status()
        return _cur_status

    def close_connection(self, stop_collect=False):
        if stop_collect:
            self.stop_collect()
        if self.cmd_controller:
            self.cmd_controller.close_conn()
            self.cmd_controller = None
        if self.data_controller:
            self.data_controller.close_conn()
            self.data_controller = None


if __name__ == '__main__':
    print("hi")
