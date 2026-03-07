import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
import time

class Preprocessor:

    def __init__(self, dataset_dict):

        # 提取数据字典和采样率
        datasets = dataset_dict['datasets']   # 格式: {0: (data, label), 1: (data, label), ...}
        self.fs = dataset_dict['fs']           # 采样率，例如 2000 Hz
        self.label_mapping = dataset_dict['label_mapping']    # label映射

        self.class_nums = len(self.label_mapping)

        self.data = {}
        for i in range(self.class_nums):
            data, _ = datasets[i]              # data shape: (n_samples, n_channels, n_timepoints)
            self.data[i] = data

    @staticmethod
    def common_average_reference(data):
        """
        全脑平均重参考：对每个样本，减去所有通道的均值。
        """
        # 沿通道轴计算均值，保持维度以便广播
        mean_chan = np.mean(data, axis=1, keepdims=True)
        return data - mean_chan

    def _design_notch_sos(self, f0, bw):
        """
        单个陷波滤波器的 SOS（二阶节）系数
        :param f0: 陷波频率 (Hz)
        :param bw: 带宽 (Hz)
        :return: sos : ndarray, shape (n_sections, 6)
        """
        Q = f0 / bw
        b, a = iirnotch(f0, Q, self.fs)
        sos = tf2sos(b, a)
        return sos

    def _design_bandpass_sos(self, lowcut, highcut, order=4):
        """
        带通滤波器的 SOS 系数
        :param lowcut: 低频截止 (Hz)
        :param highcut: 高频截止 (Hz)
        :param order: 滤波器阶数
        :return: sos : ndarray, shape (n_sections, 6)
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos

    def preprocess(self, f0_list, lowcut=1, highcut=200, bw=2, bandpass_order=4):
        """
        对每个类别的数据进行预处理：陷波 → 带通 → 共平均参考
        :param f0_list: 要滤除的工频频率列表（例如 [50, 100, 150]）
        :param lowcut: 带通下限频率
        :param highcut: 带通上限频率
        :param bw: 陷波滤波器带宽
        :param bandpass_order: 带通滤波器阶数
        :return: preprocessed_datasets
        """

        start = time.time()

        # 1. 陷波滤波器（将所有陷波 SOS 垂直拼接）
        notch_sos_list = [self._design_notch_sos(f, bw) for f in f0_list]
        if notch_sos_list:
            total_notch_sos = np.vstack(notch_sos_list)
        else:
            total_notch_sos = None   # 无陷波

        # 2. 带通滤波器 SOS
        bp_sos = self._design_bandpass_sos(lowcut, highcut, order=bandpass_order)

        # 3. 合并滤波器：先陷波后带通（SOS 级联）
        if total_notch_sos is not None:
            combined_sos = np.vstack([total_notch_sos, bp_sos])
        else:
            combined_sos = bp_sos

        preprocessed_datasets = {}

        for i in range(self.class_nums):
            data = self.data[i]   # shape (n_samples, n_channels, n_timepoints)
            print(f"Before filtering: {data.shape}")
            # 应用组合滤波器（自动沿最后一维滤波）
            if combined_sos is not None:
                filtered = sosfiltfilt(combined_sos, data, axis=-1)
            else:
                filtered = data
            print(f"After filtering: {filtered.shape}")
            # 共平均参考
            car_data = self.common_average_reference(filtered)

            preprocessed_datasets[i] = car_data
            print(f"After CAR: {car_data.shape}")

        end = time.time()
        print(f"Preprocess time: {end - start:.2f}s")
        return preprocessed_datasets