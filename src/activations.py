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
        return 1.0/(1.0+np.exp(-x))

    def deriv(self, x):
        y = self.val(x)
        return y*(1-y)

    def deriv2nd(self, x):
        y = self.val(x)
        dy = y*(1-y)
        return dy*(1-(2*y))

class Softmax(Activation):
    @staticmethod
    def softmax(vec):
        shape = vec.shape
        dims = sum([1 if i>1 else 0 for i in shape])
        if dims == 1:
            e = np.exp(vec)
            return e / np.sum(e)
        if dims == 2:
            m = np.zeros(vec.shape)
            for i in range(shape[0]):
                e = np.exp(vec[i, :])
                m[i, :] = e / np.sum(e)
            return m

    def __init__(self):
        super(Softmax, self).__init__()

    def val(self, v):
        exp = np.exp(v)
        return exp / np.sum(exp)

    def deriv(self, x):
        y = self.val(x)
        return y*(1-y)


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