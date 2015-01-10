"""
Network loss functions
"""
import math_interface as np
import math_funcs

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
        return math_funcs.softmax(vec, axis=1)

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