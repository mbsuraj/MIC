import numpy as np

def _to_array(data):
    """Convert pandas Series/DataFrame or numpy array to flattened numpy array"""
    if hasattr(data, 'values'):
        return data.values.flatten()
    return np.array(data).flatten()

def mean_absolute_percentage_error(y, yhat):
    return np.mean(abs(y-yhat) / y)

def smape(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denominator != 0, numerator / denominator, 0))
    return smape

def cumulative_error_percentage(y_true, y_pred):
    y_true, y_pred = _to_array(y_true).sum(), _to_array(y_pred).sum()
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true)
    return np.mean(numerator / denominator)


# Weighted MAPE (more interpretable)
def weighted_mape(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    weights = np.abs(y_true)
    errors = np.abs(y_true - y_pred) / np.abs(y_true)
    return round(np.average(errors, weights=weights), 2)

# Value-weighted error (cleaner)
def value_weighted_error(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return np.sum(np.abs(y_true - y_pred) * np.abs(y_true)) / np.sum(y_true ** 2)

def error_std(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return np.std(y_true - y_pred)