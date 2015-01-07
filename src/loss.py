"""
Network loss functions
"""
import numpy as np

__author__ = 'justin'


class Loss(object):
    """
    Superclass for all loss functions
    """
    def eval(self, predicted, labels):
        raise NotImplemented

    def backward(self, predicted, labels):
        raise NotImplemented


class Softmax(Loss):
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

    def eval(self, predicted, labels):
        predicted = Softmax.softmax(predicted)
        e = -labels*np.log(predicted)
        return e

    def backward(self, predicted, labels):
        predicted = Softmax.softmax(predicted)
        return predicted-labels


class Squared(Loss):
    def __init__(self):
        super(Squared, self).__init__()

    def eval(self, predicted, labels):
        diff = predicted - labels
        diff = np.linalg.norm(diff)
        e = 0.5*diff*diff
        return e

    def backward(self, predicted, labels):
        return predicted-labels