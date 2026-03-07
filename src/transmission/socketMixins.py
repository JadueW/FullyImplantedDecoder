# -*- coding:utf-8 -*-
# author:70706
# datetime:2024/11/27 17:12
# software: PyCharm
import socket


class SocketMixin:
    def connect(self):
        """
        建立连接
        :return:
        """
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port))
        return self.client

    def disconnect(self):
        """
        关闭连接
        :return: None
        """
        self.client.close()

    def set_timeout(self, value: float | None = 1.):
        """
        设置阻塞超时
        :param value:
        :return:
        """
        self.client.settimeout(value)

    def set_blocking(self, is_blocking=True):
        """
        默认为阻塞模式
        :param is_blocking:
        :return:
        """
        self.client.setblocking(is_blocking)


if __name__ == '__main__':
    print("hi")
