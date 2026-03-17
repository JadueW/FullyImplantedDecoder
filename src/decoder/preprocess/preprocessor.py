import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from functools import lru_cache

def normalize_preprocess_config(preprocess_config):
    if 'preprocess' in preprocess_config:
        preprocess_config = preprocess_config['preprocess']

    normalized = dict(preprocess_config)
    if 'bandpass_low' not in normalized and 'lowcut' in normalized:
        normalized['bandpass_low'] = normalized['lowcut']
    if 'bandpass_high' not in normalized and 'highcut' in normalized:
        normalized['bandpass_high'] = normalized['highcut']

    normalized.setdefault('notch_freqs', [])
    normalized.setdefault('notch_bandwidth', 2.0)
    normalized.setdefault('bandpass_order', 4)
    normalized.setdefault('use_car', False)
    return normalized


def _config_to_key(preprocess_config):
    normalized = normalize_preprocess_config(preprocess_config)
    return (
        tuple(normalized['notch_freqs']),
        normalized['notch_bandwidth'],
        normalized['bandpass_low'],
        normalized['bandpass_high'],
        normalized['bandpass_order']
    )

@lru_cache(maxsize=32)
def _build_filter_sos_cached(fs,config_key):
    """ 创建滤波相关缓存 """
    notch_freqs, bandwidth, lowcut, highcut, order = config_key

    notch_parts = [
        design_notch_sos(freq, bandwidth, fs)
        for freq in notch_freqs
    ]

    bandpass_sos = design_bandpass_sos(lowcut, highcut, fs, order)

    if notch_parts:
        return np.vstack(notch_parts + [bandpass_sos])
    return bandpass_sos


def common_average_reference(data_array):
    mean_chan = np.mean(data_array, axis=-2, keepdims=True)
    return data_array - mean_chan


def design_notch_sos(freq, bandwidth, fs):
    q_value = float(freq) / float(bandwidth)
    b_coef, a_coef = iirnotch(freq, q_value, fs)
    return tf2sos(b_coef, a_coef)


def design_bandpass_sos(lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = float(lowcut) / nyquist
    high = float(highcut) / nyquist
    return butter(order, [low, high], btype='band', output='sos')


def _build_filter_sos(fs, preprocess_config):
    preprocess_config = normalize_preprocess_config(preprocess_config)
    notch_freqs = preprocess_config['notch_freqs']
    bandwidth = preprocess_config['notch_bandwidth']
    lowcut = preprocess_config['bandpass_low']
    highcut = preprocess_config['bandpass_high']
    order = preprocess_config['bandpass_order']

    notch_parts = [
        design_notch_sos(freq, bandwidth, fs)
        for freq in notch_freqs
    ]
    bandpass_sos = design_bandpass_sos(lowcut, highcut, fs, order)

    if notch_parts:
        return np.vstack(notch_parts + [bandpass_sos])
    return bandpass_sos


def preprocess_data(data, fs, preprocess_config):

    preprocess_config = normalize_preprocess_config(preprocess_config)
    data = np.asarray(data, dtype=float)

    if data.ndim not in (2, 3):
        raise ValueError(
            "Expected `data` with shape (n_channels, n_timepoints) "
            "or (n_windows, n_channels, n_timepoints)."
        )

    config_key = _config_to_key(preprocess_config)
    total_sos = _build_filter_sos_cached(fs, config_key)
    filtered = sosfiltfilt(total_sos, data, axis=-1)

    if preprocess_config.get('use_car', False):
        filtered = common_average_reference(filtered)

    return filtered

