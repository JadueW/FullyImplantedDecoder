from functools import lru_cache

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
        'bands': dict(feature_config['bands']),
    }


def _feature_config_to_key(feature_config):
    feature_config = normalize_feature_config(feature_config)
    return (
        feature_config['nperseg'],
        feature_config['noverlap'],
        feature_config.get('nfft'),
        tuple(feature_config['total_power_range']),
        bool(feature_config.get('use_log_abs_power', True)),
        tuple(feature_config['feature_order']),
        tuple(
            (band_name, tuple(band_range))
            for band_name, band_range in sorted(feature_config['bands'].items())
        ),
    )


def _feature_config_from_key(config_key):
    (
        nperseg,
        noverlap,
        nfft,
        total_power_range,
        use_log_abs_power,
        feature_order,
        bands,
    ) = config_key
    return {
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft,
        'total_power_range': list(total_power_range),
        'use_log_abs_power': use_log_abs_power,
        'feature_order': list(feature_order),
        'bands': {
            band_name: list(band_range)
            for band_name, band_range in bands
        },
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


@lru_cache(maxsize=32)
def _build_feature_plan_cached(fs, n_channels, config_key):
    feature_config = _feature_config_from_key(config_key)
    nperseg = feature_config['nperseg']
    noverlap = feature_config['noverlap']
    nfft = feature_config.get('nfft')

    # Welch 频率轴仅由参数决定，适合缓存成可复用 plan。
    dummy_sample = np.zeros((n_channels, nperseg), dtype=float)
    freqs, _ = welch(
        dummy_sample,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=-1,
    )
    masks, total_mask = _prepare_masks(freqs, feature_config)

    return {
        'config': feature_config,
        'masks': masks,
        'total_mask': total_mask,
        'layout': build_feature_layout(feature_config, n_channels),
    }


def prepare_feature_plan(fs, feature_config, n_channels):
    config_key = _feature_config_to_key(feature_config)
    return _build_feature_plan_cached(float(fs), int(n_channels), config_key)


def _compute_sample_features(sample, fs, feature_plan):
    feature_config = feature_plan['config']
    _, psd = welch(
        sample,
        fs=fs,
        nperseg=feature_config['nperseg'],
        noverlap=feature_config['noverlap'],
        nfft=feature_config.get('nfft'),
        axis=-1,
    )
    eps = np.finfo(float).eps
    total_power = _safe_band_mean(psd, feature_plan['total_mask'])
    total_power = np.maximum(total_power, eps)

    beta_power = _safe_band_mean(psd, feature_plan['masks']['beta'])
    high_gamma_power = _safe_band_mean(psd, feature_plan['masks']['high_gamma'])
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
        'high_gamma_rel_psd': high_gamma_rel,
    }
    return np.concatenate([block_map[name] for name in feature_config['feature_order']])


def extract_feature(data, fs, feature_config=None, return_metadata=False, feature_plan=None):
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
    if feature_plan is None:
        if feature_config is None:
            raise ValueError("feature_config is required when feature_plan is not provided.")
        feature_plan = prepare_feature_plan(fs, feature_config, n_channels)

    feature_rows = [
        _compute_sample_features(sample, fs, feature_plan)
        for sample in samples
    ]
    feature_array = np.asarray(feature_rows, dtype=float)
    feature_layout = feature_plan['layout']

    if squeeze_output:
        feature_array = feature_array[0]

    if return_metadata:
        return feature_array, feature_layout

    return feature_array
