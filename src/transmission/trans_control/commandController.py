# -*- coding:utf-8 -*-
# author:70706
# datetime:2024/11/27 17:10
# software: PyCharm
import socket
import warnings

from transmission.socketMixins import SocketMixin


class CmdController(SocketMixin):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None

        self.cmd_buffer_size = 4096
        # 设置说明信息
        self.cmd_set_notion = lambda info: f"set notion {info}".encode('utf-8')
        self.cmd_return_expected_notion = "Return: Notion "  # 成功 True or False
        # 运行并保存命令
        self.cmd_set_record = b'set runmode record'
        # 停止运行命令
        self.cmd_set_stop = b'set runmode stop'
        # 获取运行状态命令
        self.cmd_get_runmode = b'get runmode'
        self.cmd_return_expected_runmode = 'Return: RunMode '
        # 设置marker命令
        self.cmd_set_marker = lambda evt_id: f'set marker {evt_id}'.encode('utf-8')
        self.cmd_get_marker = b'get marker'  # Return: Marker [[id, ref_time, ts], [id, ref_time, ts]]
        self.cmd_return_expected_marker = "Return: Marker "  # 如果是set marker命令， 返回当前marker的list[id, ref_time, ts] Return: Marker [id, ref_time, ts]
        # 获取采集率fs命令
        self.cmd_get_fs = b'get sampleratehertz'
        self.cmd_return_expected_fs = 'Return: SampleRateHertz '
        # 获取从set runmode record命令后的持续运行时长命令
        self.cmd_get_duration = b'get duration'
        self.cmd_return_expected_duration = 'Return: Duration '
        # 获取通道信息
        self.cmd_get_ch_info = b'get channelinfo'
        # e.g. Return: ChannelInfo [A,B]-[32,128]
        self.cmd_return_expected_ch_info = 'Return: ChannelInfo '

    def start(self):
        self.client.sendall(self.cmd_set_record)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed starting server"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_runmode) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_set_record}.'
                )
                return _cmd_rtn
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_runmode):]
            if _expected_rs.lower() != "record":
                warnings.warn(
                    f'unexpected runmode after {self.cmd_set_record}, expected record, but got {_expected_rs}.'
                )
            return _expected_rs
        else:
            return ''

    def set_notion(self, notion):
        self.client.sendall(self.cmd_set_notion(notion))
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to send notion."
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_notion) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_set_notion}.'
                )
                return _cmd_rtn
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_marker):]
            if _expected_rs.isalpha():
                if _expected_rs.capitalize() != 'True':
                    warnings.warn(
                        f'failed to set notion'
                    )
            else:
                warnings.warn(
                    f"Expected return value was True or False, but got {_expected_rs}."
                )
            return _expected_rs
        else:
            return ''

    def stop(self):
        self.client.sendall(self.cmd_set_stop)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to stop the server"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_runmode) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_set_stop}.'
                )
                return _cmd_rtn
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_runmode):]
            if _expected_rs.lower() != "stop":
                warnings.warn(
                    f'unexpected runmode after {self.cmd_set_stop}, expected stop, but got {_expected_rs}.'
                )
            return _expected_rs

        else:
            return ''

    def close_conn(self):
        if self.client:
            self.client.close()
            print("Cmd Server Connection Closed...")
        else:
            warnings.warn(
                "Please build connection first."
            )

    def get_fs(self):
        self.client.sendall(self.cmd_get_fs)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to get fs"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_fs) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_get_fs}.'
                )
                return _cmd_rtn
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_fs):]
            return int(str(_expected_rs).strip())
        else:
            return 0

    def get_marker(self):
        self.client.sendall(self.cmd_get_marker)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to get marker"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_marker) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_get_marker}.'
                )
                return []
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_marker):]
            try:
                return eval(_expected_rs.strip())
            except Exception as e:
                print(e)
                print(f"Current {len(_cmd_rtn)}")
                warnings.warn(f"May the length of all marker exceed {self.cmd_buffer_size}")
                return []

        else:
            return []

    def set_marker(self, event: int = 1):
        if event <= 0:
            event = 0
        self.client.sendall(self.cmd_set_marker(str(event)))
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to set marker"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_marker) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_set_marker(str(event))}.'
                )
                return []
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_marker):]
            try:
                return eval(_expected_rs.strip())
            except Exception as e:
                print(e)
                print(f"Current {len(_cmd_rtn)}")
                warnings.warn(f"May the length of all marker exceed {self.cmd_buffer_size}")
                return []
        else:
            return []

    def get_channel_info(self):
        self.client.sendall(self.cmd_get_ch_info)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to get channel information"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_ch_info) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_get_ch_info}.'
                )
                return _cmd_rtn, _cmd_rtn
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_ch_info):]
            ports, num_ch_per_port = str(_expected_rs).split('-')
            ports = ports.strip()
            ports = ports[1:-1].split(',')
            return ports, eval(str(num_ch_per_port).strip())
        else:
            return None, None

    def get_num_channel(self):
        _ports, _num_ch_per_port = self.get_channel_info()
        if _num_ch_per_port:
            return sum(_num_ch_per_port)
        else:
            return 0

    def get_duration(self):
        self.client.sendall(self.cmd_get_duration)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to get duration"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_duration) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_get_duration}.'
                )
                return 0
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_duration):]
            return float(str(_expected_rs).strip())
        else:
            return 0

    def get_num_marker(self):
        _marker = self.get_marker()
        if _marker:
            return len(_marker)
        else:
            return 0

    def get_status(self):
        self.client.sendall(self.cmd_get_runmode)
        try:
            _cmd_rtn = str(self.client.recv(self.cmd_buffer_size), "utf-8")
            is_success = True
        except TimeoutError:
            is_success = False
            warnings.warn(
                "failed to get status"
            )

        if is_success:
            if _cmd_rtn.find(self.cmd_return_expected_runmode) == -1:
                warnings.warn(
                    f'failed to receive return value after {self.cmd_get_runmode}.'
                )
                return _cmd_rtn
            _expected_rs = _cmd_rtn[len(self.cmd_return_expected_runmode):]
            return str(_expected_rs).strip()
        else:
            return ''


if __name__ == '__main__':
    print("hi")
