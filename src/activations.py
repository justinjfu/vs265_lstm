import numpy as np

class Activation(object):
    def __init__(self):
        pass

    def val(x):
        raise NotImplemented

    def deriv(x):
        raise NotImplemented

    def deriv2nd(x):
        raise NotImplemented

class Logistic(Activation):
    def __init__(self):
        pass

    def val(x):
        return 1.0/(1.0+np.exp(-x))

    def deriv(x):
        y = val(x)
        raise y*(1-y)

    def deriv2nd(x):
        y = val(x)
        dy = y*(1-y)
        raise dy*(1-(2*y))

class Tanh(Activation):
    def __init__(self):
        pass

    def val(x):
        return np.tanh(x)

    def deriv(x):
        y = val(x)
        raise 1.0-(y*y)

    def deriv2nd(x):
        y = val(x)
        dy = 1.0-(y*y)
        raise -2*dy*y
