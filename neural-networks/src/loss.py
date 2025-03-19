import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_gradient(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_true.size

def cross_entropy_loss(y_true, y_pred):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))

def cross_entropy_loss_gradient(y_true, y_pred):
    # Clip predictions to prevent division by zero
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - (y_true / y_pred) / y_true.shape[0]

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
