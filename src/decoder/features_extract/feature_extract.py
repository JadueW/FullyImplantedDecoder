import numpy as np
from scipy.signal import welch
from joblib import Parallel, delayed   # 可选并行，需要安装 joblib

class FeatureExtractor:
    """
        频带功率特征提取器（绝对功率 + 相对功率）
    """

    def __init__(self, fs=2000, bands=None, nperseg=2048, noverlap=1024):
        """
        :param fs: 采样率
        :param bands: 段名称与频率范围，默认包含 delta, theta, alpha, beta, low_gamma, high_gamma
        :param nperseg:  Welch 方法的每段长度
        :param noverlap: Welch 方法的重叠样本数
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

        if bands is None:
            self.bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'low_gamma': (30, 70),
                'high_gamma': (70, 150)
            }
        else:
            self.bands = bands

        # 频段名称列表，保持顺序
        self.band_names = list(self.bands.keys())
        self.n_bands = len(self.band_names)

        # 预计算频率点和频带掩码（将在第一次调用 compute_psd_features 时计算）
        self._freqs = None
        self._band_masks = None

    def _compute_frequency_masks(self):
        """
        计算 Welch 方法对应的频率点和各频段的布尔掩码。
        假设使用固定的 fs, nperseg，频率点不变。
        """
        # 计算一次频率点（任意数据长度相同，取 1 个样本的通道数即可）
        dummy_data = np.zeros((1, 1, self.nperseg * 2))  # 确保长度足够
        freqs, _ = welch(dummy_data[0], fs=self.fs, nperseg=self.nperseg,
                         noverlap=self.noverlap, axis=-1)
        self._freqs = freqs

        # 为每个频段创建布尔掩码
        self._band_masks = {}
        for band_name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs <= high)
            self._band_masks[band_name] = mask

    def compute_psd_features(self, data, parallel=True, n_jobs=-1):
        """
        从数据中提取频带绝对功率和相对功率特征。
        :param data: 预处理后的数据，shape (n_samples, n_channels, n_timepoints)
        :param parallel: 是否使用并行处理
        :param n_jobs: 并行任务数，-1 表示使用所有 CPU 核心
        :return: shape (n_samples, n_channels * n_bands * 2)，顺序为：通道1绝对功率(频段1..n)，通道2绝对功率...，通道1相对功率...
        """
        n_samples, n_channels, n_timepoints = data.shape

        # 如果尚未计算频率掩码，则计算
        if self._freqs is None:
            self._compute_frequency_masks()

        # 准备存储特征的数组
        abs_features = np.zeros((n_samples, n_channels, self.n_bands))
        rel_features = np.zeros((n_samples, n_channels, self.n_bands))

        # 定义处理单个样本的函数（便于并行）
        def process_sample(sample_idx):
            # 计算 PSD，形状 (n_channels, n_freqs)
            _, psd = welch(data[sample_idx], fs=self.fs,
                           nperseg=self.nperseg, noverlap=self.noverlap, axis=-1)

            # 总功率 (1–150 Hz) 使用对数尺度
            total_mask = (self._freqs >= 1) & (self._freqs <= 150)
            total_power = 10 * np.log10(np.mean(psd[:, total_mask], axis=1))  # (n_channels,)

            # 初始化当前样本的特征
            abs_row = np.zeros((n_channels, self.n_bands))
            rel_row = np.zeros((n_channels, self.n_bands))

            for j, band_name in enumerate(self.band_names):
                mask = self._band_masks[band_name]
                band_power = np.mean(psd[:, mask], axis=1)           # (n_channels,)
                abs_row[:, j] = band_power
                rel_row[:, j] = band_power / (10**(total_power/10))  # 相对功率（线性比例）

            return abs_row, rel_row

        # 并行或串行执行
        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_sample)(i) for i in range(n_samples)
            )
            for i, (abs_row, rel_row) in enumerate(results):
                abs_features[i] = abs_row
                rel_features[i] = rel_row
        else:
            for i in range(n_samples):
                abs_features[i], rel_features[i] = process_sample(i)

        # 将特征展平并拼接：绝对功率 + 相对功率
        abs_flat = abs_features.reshape(n_samples, -1)
        rel_flat = rel_features.reshape(n_samples, -1)
        features = np.concatenate([abs_flat, rel_flat], axis=1)

        return features

    def extract_features_and_labels(self, processed_datasets, labels=None):
        """
        从预处理后的数据集中提取特征，并返回特征矩阵和对应的标签向量。
        :param processed_datasets:键为类别索引，值为预处理后的数据，形状 (n_samples, n_channels, n_timepoints)
        :param labels:类别标签列表，用于构建标签向量。如果为 None，则使用字典的键作为标签。
        :return: features_dict 键为类别索引，值为该类别的特征矩阵，形状 (n_samples, n_features)
        """
        features_dict = {}
        if labels is None:
            labels = list(processed_datasets.keys())
        for class_idx in labels:
            data = processed_datasets[class_idx]
            features = self.compute_psd_features(data)
            features_dict[class_idx] = features
        return features_dict