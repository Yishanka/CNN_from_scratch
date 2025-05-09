from cnn.core.tensor import Tensor
from cnn.core.parameter import Parameter
import numpy as np
        
def xavier_uniform(param: Parameter, fan_in, fan_out):
    a = np.sqrt(6.0 / (fan_in + fan_out))
    param.data = np.random.uniform(-a, a, size=param.data.shape)

def xavier_normal(param: Parameter, fan_in, fan_out):
    sigma = np.sqrt(2.0 / (fan_in + fan_out))
    param.data = np.random.normal(0, sigma, size=param.data.shape)

def he_uniform(param: Parameter, fan_in):
    a = np.sqrt(6.0 / fan_in)
    param.data = np.random.uniform(-a, a, size=param.data.shape)

def he_normal(param: Parameter, fan_in):
    sigma = np.sqrt(2.0 / fan_in)
    param.data = np.random.normal(0, sigma, size=param.data.shape)
