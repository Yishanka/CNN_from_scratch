import numpy as np
from numpy import ndarray

def xavier_uniform(n_in: int, n_out: int)->ndarray:
    a = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-a, a, size=(n_out,n_in))

def xavier_normal(n_in: int, n_out: int)->ndarray:
    sigma = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0, sigma, size=(n_out, n_in))

def he_uniform(n_in: int, n_out: int)->ndarray:
    a = np.sqrt(6.0 / n_in)
    return np.random.uniform(-a, a, size=(n_out, n_in))

def he_normal(n_in: int, n_out: int)->ndarray:
    sigma = np.sqrt(2.0 / n_in)
    return np.random.normal(0, sigma, size=(n_out, n_in))

def zeros(n_in: int, n_out: int)->ndarray:
    return np.zeros((n_out, n_in))