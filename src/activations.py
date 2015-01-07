"""
Network non-linearities and activations functions
"""
import math_interface as np


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


class Identity(Activation):
    def __init__(self):
        super(Identity, self).__init__()

    def val(self, x):
        return x

    def deriv(self, x):
        return np.ones(x.shape)

    def deriv2nd(self, x):
        y = self.val(x)
        dy = y*(1-y)
        return dy*(1-(2*y))


class Logistic(Activation):
    def __init__(self):
        super(Logistic, self).__init__()

    def val(self, x):
        return np.logistic(x)

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


class Softplus(Activation):
    def __init__(self):
        super(Softplus, self).__init__()

    def val(self, x):
        return np.log(1.0+np.exp(x))

    def deriv(self, x):
        return np.logistic(x)

    def deriv2nd(self, x):
        y = self.deriv(x)
        return y*(1-y)