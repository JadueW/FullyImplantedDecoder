# -*- coding:utf-8 -*-
# author:70706
# datetime:2024/12/24 15:57
# software: PyCharm
import sys
import os
import time
import warnings
import serial

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transmission.trans_manager.remoteManager import RemoteManager

class RemoteManagerTTL(RemoteManager):
    def __init__(self, com_port="COM8", baudrate=115200, **kwargs):
        super().__init__(**kwargs)
        print(com_port, baudrate)
        self.serial_port = serial.Serial(com_port, baudrate, timeout=1)
        print("Serial connected...")
        self.marker = []
        self.dig_in_state: int = 0

    def __clear_bit(self, bit_number):
        # 创建掩码：掩码是255 - (1 << (bit_number - 1))
        mask = 255 - (1 << (bit_number - 1))
        # 执行位与操作：将原始字节与掩码进行与操作
        return self.dig_in_state & mask

    def __set_bit(self, bit_number):
        # 创建掩码：掩码是1 << (bit_number - 1)
        mask = 1 << (bit_number - 1)
        # 执行位或操作：将原始字节与掩码进行或操作
        return self.dig_in_state | mask

    def set_bit_marker(self, marker_id: int = 1):
        """
        走非tcp逻辑, 只能设置1-8,和设置10- 80用于拉低
        examples:
            set_marker(1)  # 拉高 1
            set_marker(8)  # 拉高 8
            set_marker(2)  # 拉高 2
            set_marker(10)  # 拉低 1
            set_marker(80)  # 拉低 8
            set_marker(20)  # 拉低 2
            set_marker(0)  # 拉低所有通道
        """
        # 检查marker_id
        set_id = [i + 1 for i in range(8)]
        clear_id = [(i + 1) * 10 for i in range(8)]
        if int(marker_id) in set_id:
            marker_bytes_value = self.__set_bit(int(marker_id))
        elif int(marker_id) in clear_id:
            _marker_id, state = str(marker_id)[0], str(marker_id)[1]
            marker_bytes_value = self.__clear_bit(int(_marker_id))
        elif int(marker_id) == 0:
            marker_bytes_value = 0
        else:
            warnings.warn(f"Failed to set marker. marker_id {marker_id} is not in [1-8] or [10-80]")
            return []

        # 将数据位更新
        self.dig_in_state = marker_bytes_value

        # 一个字节不涉及大端或小端序，
        bytes_value = self.dig_in_state.to_bytes(1, byteorder="big")
        self.serial_port.write(bytes_value)
        self.num_marker += 1

        # 手动计算运行时长并存入marker
        ref_time = time.time() - self.startup_time
        ts = ref_time * self.fs
        cur_marker = [marker_id, round(ref_time, 6), int(ts)]
        self.marker.append(cur_marker)
        return cur_marker

    def set_byte_marker(self, marker_id: int = 1):
        """
        走非tcp逻辑, 用于自定义marker
        marker id 可以为0-255任意整数，其对应的位的1或0标志相应dig in 通道的高低

        examples:
            set_custom_marker(marker_id=0b00000000) 将所有通道拉低
            set_custom_marker(marker_id=0b11111111) 将所有通道拉高
            set_custom_marker(marker_id=0b00010001) 将第1位和第5位拉高，其它为低
        """
        try:
            marker_id = int(marker_id)
        except ValueError:
            warnings.warn(f"Failed to set marker. Expected marker_id a integer type of 0-255, got a {type(marker_id)}")
            return []

        if 0 <= int(marker_id) <= 255:
            # 在该用法中, 用户所看见的即为其所设置的
            self.dig_in_state = marker_id

            # 一个字节不涉及大端或小端序，
            bytes_value = self.dig_in_state.to_bytes(1, byteorder="big")
            self.serial_port.write(bytes_value)
            self.num_marker += 1

            # 手动计算运行时长并存入marker
            ref_time = time.time() - self.startup_time
            ts = ref_time * self.fs
            cur_marker = [marker_id, round(ref_time, 6), int(ts)]
            self.marker.append(cur_marker)
            return cur_marker
        else:
            warnings.warn(f"Failed to set marker. Expected marker_id {marker_id} a integer of 0-255")
            return []

    def set_marker(self, marker_id: int | str = 1, mode="bit"):
        """
        bit mode, marker_id 1-8 拉高 10-80 拉低
        byte mode marker_id 0b11110000 分别为高低位
        """
        if mode.lower() == "bit":
            if isinstance(marker_id, int):
                return self.set_bit_marker(marker_id)
            else:
                warnings.warn(f"Failed to set marker_id {marker_id}, expected a integer, but got {type(marker_id)}")
        else:
            if isinstance(marker_id, str):
                marker_id = marker_id.lower()
                if marker_id.startswith("0b"):
                    return self.set_byte_marker(eval(marker_id))
                else:
                    warnings.warn(f"Failed to set marker_id {marker_id}. Marker_id should start with 0b.")
            else:
                warnings.warn(f"Failed to set marker_id {marker_id}. Expected a string, but got {type(marker_id)}")

    def get_marker(self):
        """
        走非tcp逻辑，所以自行统计
        """

        return self.marker

    def close_connection(self, stop_collect=True):
        if stop_collect:
            self.stop_collect()
        if self.cmd_controller:
            self.cmd_controller.close_conn()
            self.cmd_controller = None
        if self.data_controller:
            self.data_controller.close_conn()
            self.data_controller = None
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None


if __name__ == '__main__':
    print("hi")
