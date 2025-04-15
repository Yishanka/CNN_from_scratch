import numpy as np
from cnn.core import Parameter

def xavier_uniform(param:Parameter)->Parameter:
    a = np.sqrt(6.0 / sum(param.data.shape))
    param.data = np.random.uniform(-a, a, size=param.data.shape)
    return param

def xavier_normal(param:Parameter)->Parameter:
    sigma = np.sqrt(2.0 / sum(param.data.shape))
    param.data = np.random.normal(0, sigma, size=param.data.shape)
    return param

def he_uniform(param:Parameter)->Parameter:
    a = np.sqrt(6.0 / param.data.shape[1])
    param.data = np.random.uniform(-a, a, size=param.data.shape)
    return param

def he_normal(param:Parameter)->Parameter:
    sigma = np.sqrt(2.0 / param.data.shape[1])
    param.data = np.random.normal(0, sigma, size=param.data.shape)
    return param
