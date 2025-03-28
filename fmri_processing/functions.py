import numpy as np

def calc_maximum(data):
    return np.max(data, axis=1)


def calc_minimum(data):
    return np.min(data, axis=1)


def calc_max_min(data):
    return np.max(data, axis=1) - np.min(data, axis=1)


def calc_auc(data):
    return np.trapz(data, axis=1)  # Интегрируем по времени для каждого ответа и региона
