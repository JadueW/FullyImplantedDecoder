import joblib
import numpy as np


def load_model(model_path):
    return joblib.load(model_path)


def _get_estimator(model_obj):
    return model_obj['model']


def _get_class_labels(model_obj, estimator):
    if isinstance(model_obj, dict):
        if 'class_labels' in model_obj and model_obj['class_labels'] is not None:
            return list(model_obj['class_labels'])
        if 'classes_' in model_obj and model_obj['classes_'] is not None:
            return list(model_obj['classes_'])
        if 'label_mapping' in model_obj and model_obj['label_mapping']:
            mapping = model_obj['label_mapping']
            if isinstance(mapping, dict):
                return [mapping[key] for key in sorted(mapping)]

    if hasattr(estimator, 'classes_'):
        return list(estimator.classes_)

    return None


def build_model(model_bundle):

    estimator = _get_estimator(model_bundle)
    class_labels = _get_class_labels(model_bundle, estimator)

    return {
        'model': estimator,
        'class_labels': class_labels,
        'raw_bundle': model_bundle
    }


def _prepare_feature_array(feature_data):
    feature_array = np.asarray(feature_data, dtype=float)

    if feature_array.ndim == 1:
        return feature_array.reshape(1, -1), True
    if feature_array.ndim == 2:
        return feature_array, False

    raise ValueError(
        "Expected feature_data with shape (n_features,) or (n_samples, n_features)."
    )


def _build_probability_dict(probabilities, class_labels):
    if class_labels is None:
        class_labels = list(range(len(probabilities)))
    return {
        str(label): float(prob)
        for label, prob in zip(class_labels, probabilities)
    }


def predict_from_features(feature_data, model_obj):

    built = build_model(model_obj)
    estimator = built['model']
    class_labels = built['class_labels']
    features, squeeze_output = _prepare_feature_array(feature_data)

    predicted_indices = estimator.predict(features)
    has_proba = hasattr(estimator, 'predict_proba')

    if has_proba:
        proba = estimator.predict_proba(features)
    else:
        proba = None

    results = []
    for idx, predicted in enumerate(predicted_indices):
        predicted_label = predicted.item() if hasattr(predicted, 'item') else predicted
        result = {
            'predicted_target': predicted_label,
            'probabilities': {},
            'confidence': None,
            'method': 'ml_decoder',
            'success': True
        }

        if proba is not None:
            prob_row = proba[idx]
            result['probabilities'] = _build_probability_dict(prob_row, class_labels)
            result['confidence'] = float(np.max(prob_row))
        else:
            result['confidence'] = 1.0

        results.append(result)

    if squeeze_output:
        return results[0]
    return results


def decode(feature_data, model_obj):

    results = predict_from_features(feature_data, model_obj)
    confidence_threshold = 0.45
    def apply_threshold(result):
        result = dict(result)
        result['success'] = result['confidence'] >= confidence_threshold
        return result

    if isinstance(results, list):
        return [apply_threshold(result) for result in results]
    return apply_threshold(results)
