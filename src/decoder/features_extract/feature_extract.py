import numpy as np
from scipy.signal import welch


def build_feature_layout(feature_config, n_channels):
    return {
        'n_channels': n_channels,
        'feature_order': list(feature_config['feature_order']),
        'bands': dict(feature_config['bands'])
    }


def _prepare_masks(freqs, feature_config):
    masks = {}
    for band_name, band_range in feature_config['bands'].items():
        low, high = band_range
        masks[band_name] = (freqs >= low) & (freqs <= high)
    total_low, total_high = feature_config['total_power_range']
    total_mask = (freqs >= total_low) & (freqs <= total_high)
    return masks, total_mask


def _compute_sample_features(sample, fs, feature_config, masks, total_mask):
    _, psd = welch(
        sample,
        fs=fs,
        nperseg=feature_config['nperseg'],
        noverlap=feature_config['noverlap'],
        axis=-1
    )
    eps = np.finfo(float).eps
    total_power = np.mean(psd[:, total_mask], axis=1)
    total_power = np.maximum(total_power, eps)

    beta_power = np.mean(psd[:, masks['beta']], axis=1)
    high_gamma_power = np.mean(psd[:, masks['high_gamma']], axis=1)
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


def extract_feature(datasets, feature_config):
    n_channels = datasets.shape[0]
    freqs, _ = welch(
        datasets,
        fs=fs,
        nperseg=feature_config['nperseg'],
        noverlap=feature_config['noverlap'],
        axis=-1
    )
    masks, total_mask = _prepare_masks(freqs, feature_config)

    feature_bundle = {
        'datasets': {},
        'label_mapping': dict(bundle['label_mapping']),
        'fs': bundle['fs'],
        'metadata': dict(bundle.get('metadata', {}))
    }
    feature_bundle['metadata']['feature_layout'] = build_feature_layout(feature_config, n_channels)

    for class_id in class_ids:
        rows = []
        data_array = np.asarray(bundle['datasets'][class_id])
        for sample in data_array:
            rows.append(_compute_sample_features(sample, bundle['fs'], feature_config, masks, total_mask))
        feature_bundle['datasets'][class_id] = np.asarray(rows, dtype=float)

    return feature_bundle
