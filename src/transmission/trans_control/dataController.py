# -*- coding:utf-8 -*-
# author:70706
# datetime:2024/11/27 17:11
# software: PyCharm
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import transmission.trans_lib.waveSocket as ws

class DataController(ws.DataRecv):
    def __init__(self, host, port, num_stream, fs):
        self.fs = fs
        self.num_ch = num_stream * 32
        super().__init__(host, port, num_stream, fs)

    def get_data(self, num_point):
        try:
            _data = self.getData(int(num_point))
            _data = _data.T
            # _data = np.reshape(_data, (self.num_ch, -1))
            return _data
        except Exception as e:
            print(f"Get data error: {e}")
            self.close_conn()
            return np.array([])

    def get_buffer_size(self):
        _bt = self.getBufferTime()
        _bs = int(_bt * self.fs)
        return _bs

    def close_conn(self):
        try:
            self.end()
            self.disConnect()
            print("Data Server Connection Closed...")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    print("hi")
