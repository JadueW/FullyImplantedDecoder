import numpy as np
from scipy.signal import welch


def normalize_feature_config(feature_config):
    if 'features' in feature_config:
        return feature_config['features']
    return feature_config


def build_feature_layout(feature_config, n_channels):
    feature_config = normalize_feature_config(feature_config)
    return {
        'n_channels': n_channels,
        'feature_order': list(feature_config['feature_order']),
        'bands': dict(feature_config['bands'])
    }


def _prepare_masks(freqs, feature_config):
    feature_config = normalize_feature_config(feature_config)
    masks = {}
    for band_name, band_range in feature_config['bands'].items():
        low, high = band_range
        masks[band_name] = (freqs >= low) & (freqs <= high)
    total_low, total_high = feature_config['total_power_range']
    total_mask = (freqs >= total_low) & (freqs <= total_high)
    return masks, total_mask


def _safe_band_mean(psd, mask):
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=float)
    return np.mean(psd[:, mask], axis=1)


def _compute_sample_features(sample, fs, feature_config, masks, total_mask):
    feature_config = normalize_feature_config(feature_config)
    _, psd = welch(
        sample,
        fs=fs,
        nperseg=feature_config['nperseg'],
        noverlap=feature_config['noverlap'],
        axis=-1
    )
    eps = np.finfo(float).eps
    total_power = _safe_band_mean(psd, total_mask)
    total_power = np.maximum(total_power, eps)

    beta_power = _safe_band_mean(psd, masks['beta'])
    high_gamma_power = _safe_band_mean(psd, masks['high_gamma'])
    beta_power = np.maximum(beta_power, eps)
    high_gamma_power = np.maximum(high_gamma_power, eps)

    if feature_config.get('use_log_abs_power', True):
        beta_abs = np.log10(beta_power)
        high_gamma_abs = np.log10(high_gamma_power)
    else:
        beta_abs = beta_power
        high_gamma_abs = high_gamma_power

    beta_rel = beta_power / total_power
    high_gamma_rel = high_gamma_power / total_power

    block_map = {
        'beta_abs_psd': beta_abs,
        'beta_rel_psd': beta_rel,
        'high_gamma_abs_psd': high_gamma_abs,
        'high_gamma_rel_psd': high_gamma_rel
    }
    return np.concatenate([block_map[name] for name in feature_config['feature_order']])


def extract_feature(data, fs, feature_config, return_metadata=False):

    feature_config = normalize_feature_config(feature_config)
    data = np.asarray(data, dtype=float)

    if data.ndim == 2:
        samples = data[np.newaxis, ...]
        squeeze_output = True
    elif data.ndim == 3:
        samples = data
        squeeze_output = False
    else:
        raise ValueError(
            "Expected `data` with shape (n_channels, n_timepoints) "
            "or (n_windows, n_channels, n_timepoints)."
        )

    n_channels = samples.shape[1]
    freqs, _ = welch(
        samples[0],
        fs=fs,
        nperseg=feature_config['nperseg'],
        noverlap=feature_config['noverlap'],
        axis=-1
    )
    masks, total_mask = _prepare_masks(freqs, feature_config)
    feature_rows = [
        _compute_sample_features(sample, fs, feature_config, masks, total_mask)
        for sample in samples
    ]
    feature_array = np.asarray(feature_rows, dtype=float)
    feature_layout = build_feature_layout(feature_config, n_channels)

    if squeeze_output:
        feature_array = feature_array[0]

    if return_metadata:
        return feature_array, feature_layout

    return feature_array
