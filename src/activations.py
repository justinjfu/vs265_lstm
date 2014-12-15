import numpy as np


class Activation(object):
    def __init__(self):
        pass

    def val(self, x):
        raise NotImplemented

    def deriv(self, x):
        raise NotImplemented

    def deriv2nd(self, x):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self.val(*args)


class Logistic(Activation):
    def __init__(self):
        super(Logistic, self).__init__()

    def val(self, x):
        return 1.0/(1.0+np.exp(-x))

    def deriv(self, x):
        y = self.val(x)
        return y*(1-y)

    def deriv2nd(self, x):
        y = self.val(x)
        dy = y*(1-y)
        return dy*(1-(2*y))


class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def val(self, x):
        return np.tanh(x)

    def deriv(self, x):
        y = self.val(x)
        return 1.0-(y*y)

    def deriv2nd(self, x):
        y = self.val(x)
        dy = 1.0-(y*y)
        return -2*dy*y