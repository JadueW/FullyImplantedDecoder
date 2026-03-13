import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

"""
    预处理工具类
"""


def common_average_reference(data_array):
    """
    CAR 全脑平均参考
    :param data_array:
    :return:
    """
    mean_chan = np.mean(data_array, axis=1, keepdims=True)
    return data_array - mean_chan


def design_notch_sos(freq, bandwidth, fs):
    """
    陷波滤波
    :param freq:
    :param bandwidth:
    :param fs:
    :return:
    """
    q_value = float(freq) / float(bandwidth)
    b_coef, a_coef = iirnotch(freq, q_value, fs)
    return tf2sos(b_coef, a_coef)


def design_bandpass_sos(lowcut, highcut, fs, order):
    """
    带通滤波
    :param lowcut:
    :param highcut:
    :param fs:
    :param order:
    :return:
    """
    nyquist = 0.5 * fs
    low = float(lowcut) / nyquist
    high = float(highcut) / nyquist
    return butter(order, [low, high], btype='band', output='sos')


def preprocess_bundle(bundle, preprocess_config):
    fs = bundle['fs']
    notch_freqs = preprocess_config['notch_freqs']
    bandwidth = preprocess_config['notch_bandwidth']
    lowcut = preprocess_config['bandpass_low']
    highcut = preprocess_config['bandpass_high']
    order = preprocess_config['bandpass_order']
    use_car = preprocess_config.get('use_car', True)

    notch_parts = []
    for freq in notch_freqs:
        notch_parts.append(design_notch_sos(freq, bandwidth, fs))
    bandpass_sos = design_bandpass_sos(lowcut, highcut, fs, order)
    if notch_parts:
        total_sos = np.vstack(notch_parts + [bandpass_sos])
    else:
        total_sos = bandpass_sos

    processed = {
        'datasets': {},
        'label_mapping': dict(bundle['label_mapping']),
        'fs': fs,
        'metadata': dict(bundle.get('metadata', {}))
    }
    processed['metadata']['preprocess'] = {
        'notch_freqs': list(notch_freqs),
        'bandpass_low': lowcut,
        'bandpass_high': highcut,
        'bandpass_order': order,
        'use_car': use_car
    }

    for class_id in sorted(bundle['datasets']):
        data_array = np.asarray(bundle['datasets'][class_id])
        filtered = sosfiltfilt(total_sos, data_array, axis=-1)
        if use_car:
            filtered = common_average_reference(filtered)
        processed['datasets'][class_id] = filtered

    return processed
