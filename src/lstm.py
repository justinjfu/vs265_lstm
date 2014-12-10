import numpy as np
from objective import Objective, Weights


class LSTMObjective(Objective):
    def __init__(self):
        super(LSTMObjective, self).__init__()

    def gradient_at(self, wts):
        raise NotImplemented

    def value_at(self, wts):
        raise NotImplemented


class LSTMWeights(Weights):
    def __init__(self):
        super(LSTMWeights, self).__init__()

    def add_weight(self, other_weight):
        raise NotImplemented

    def mul_scalar(self, other_scalar):
        raise NotImplemented

    def save_to_file(self, filename):
        raise NotImplemented

    def __str__(self):
        return str(self.val)

