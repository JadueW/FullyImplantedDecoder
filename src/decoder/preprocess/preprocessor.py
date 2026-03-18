from functools import lru_cache

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos


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
    normalized.setdefault('downsample_fs', None)
    normalized.setdefault('downsample_order', 4)
    return normalized


def _config_to_key(preprocess_config):
    normalized = normalize_preprocess_config(preprocess_config)
    return (
        tuple(normalized['notch_freqs']),
        normalized['notch_bandwidth'],
        normalized['bandpass_low'],
        normalized['bandpass_high'],
        normalized['bandpass_order'],
        normalized['downsample_fs'],
        normalized['downsample_order'],
    )


def get_preprocess_output_fs(fs, preprocess_config):
    normalized = normalize_preprocess_config(preprocess_config)
    down_fs = normalized.get('downsample_fs')
    if down_fs is None:
        return float(fs)
    down_fs = float(down_fs)
    if down_fs <= 0:
        raise ValueError('downsample_fs must be positive when provided.')
    if down_fs > fs:
        raise ValueError('downsample_fs should not be greater than fs.')
    return down_fs


@lru_cache(maxsize=32)
def _build_filter_sos_cached(fs, config_key):
    notch_freqs, bandwidth, lowcut, highcut, order, _, _ = config_key
    notch_parts = [design_notch_sos(freq, bandwidth, fs) for freq in notch_freqs]
    bandpass_sos = design_bandpass_sos(lowcut, highcut, fs, order)
    return np.vstack(notch_parts + [bandpass_sos]) if notch_parts else bandpass_sos


@lru_cache(maxsize=16)
def _build_downsample_sos_cached(fs, down_fs, order):
    cutoff = 0.8 * (float(down_fs) / 2.0)
    return butter(order, cutoff, btype='low', fs=fs, output='sos')


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


def down_sampling(data_array, fs, down_fs, order=4):
    data_array = np.asarray(data_array, dtype=float)
    fs = float(fs)
    down_fs = float(down_fs)

    if down_fs <= 0 or fs <= 0:
        raise ValueError('fs and down_fs must be positive.')
    if down_fs > fs:
        raise ValueError('down_fs should not be greater than fs.')
    if np.isclose(down_fs, fs):
        return data_array

    ratio = fs / down_fs
    factor = int(round(ratio))

    if np.isclose(ratio, factor):
        downsample_sos = _build_downsample_sos_cached(fs, down_fs, int(order))
        filtered = sosfiltfilt(downsample_sos, data_array, axis=-1)
        return filtered[..., ::factor]

    return resample_poly(data_array, int(round(down_fs)), int(round(fs)), axis=-1)


def preprocess_data(data, fs, preprocess_config, return_fs=False):
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

    output_fs = get_preprocess_output_fs(fs, preprocess_config)
    if not np.isclose(output_fs, fs):
        filtered = down_sampling(
            filtered,
            fs=fs,
            down_fs=output_fs,
            order=preprocess_config.get('downsample_order', 4),
        )

    if return_fs:
        return filtered, output_fs
    return filtered
