from cnn.core.tensor import Tensor
from cnn.core.parameter import Parameter
from cnn.core.moniter import LossMonitor, MetricMonitor
import numpy as np
        
def xavier_uniform(param: Parameter):
    a = np.sqrt(6.0 / sum(param._data.shape))
    param._data = np.random.uniform(-a, a, size=param._data.shape)

def xavier_normal(param: Parameter):
    sigma = np.sqrt(2.0 / sum(param._data.shape))
    param._data = np.random.normal(0, sigma, size=param._data.shape)

def he_uniform(param: Parameter):
    a = np.sqrt(6.0 / param._data.shape[1])
    param._data = np.random.uniform(-a, a, size=param._data.shape)

def he_normal(param: Parameter):
    sigma = np.sqrt(2.0 / param._data.shape[1])
    param._data = np.random.normal(0, sigma, size=param._data.shape)
