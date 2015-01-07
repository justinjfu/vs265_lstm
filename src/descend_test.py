"""
Simple test on the gradient descent code.

Tries to minimize f(x) = x^2+3
"""
from objective import Objective, Weights
from descend import gd


class SimpleObjective(Objective):
    def __init__(self):
        super(SimpleObjective, self).__init__()

    def get_init_wt(self):
        return SimpleWeight(10)

    def gradient_at(self, wts):
        return SimpleWeight(2*wts.val)

    def value_at(self, wts):
        return wts.val**2+3


class SimpleWeight(Weights):
    def __init__(self, val):
        super(SimpleWeight, self).__init__()
        self.val = val

    def add_weight(self, other_weight):
        return SimpleWeight(self.val+other_weight.val)

    def mul_scalar(self, other_scalar):
        return SimpleWeight(self.val*other_scalar)

    def save_to_file(self, filename, i):
        print 'Saving weight ', self.val, ' to file: ', filename, ' on iter ', i

    def __str__(self):
        return str(self.val)

obj = SimpleObjective()
wt = gd(obj, obj.get_init_wt(), save_to_file='test_file', iters=200, heartbeat = 20)
print 'Final weight: ', wt, ', Objective: ', obj.value_at(wt)
