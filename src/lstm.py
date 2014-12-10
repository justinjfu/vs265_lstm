import numpy as np
from objective import Objective, Weights


class LSTMObjective(Objective):
    def __init__(self, training_set):
        super(LSTMObjective, self).__init__()

    def gradient_at(self, wts):
        raise NotImplemented

    def value_at(self, wts):
        raise NotImplemented


class LSTMWeights(Weights):
    def __init__(self):
        super(LSTMWeights, self).__init__()
        self.layers = []

    def add_weight(self, other_weight):
        raise NotImplemented

    def mul_scalar(self, other_scalar):
        raise NotImplemented

    def save_to_file(self, filename):
        raise NotImplemented

    def __str__(self):
        raise NotImplemented

#Feedforward pass
class LSTMLayerActivations(object):
    def __init__(self, n):
        pass

    def forward(self, weights, prev_cell, input):
        pass

class LSTMLayerWeights(object):
    def __init__(self, n, n_input, act_f, act_g, act_h):
        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        self.forgetw_x = np.zeros(n, n_input)  # forget weights from X
        self.forgetw_i = np.zeros(n, n_input)  # forget weights from previous hidden
        self.forgetw_c = np.zeros(n, n)  # forget weights from previous cell state

        self.inw_x = np.zeros(n, n_input)  # input weights from X
        self.inw_i = np.zeros(n, n_input)  # input weights from previous hidden
        self.inw_c = np.zeros(n, n)  # input weights from previous cell state

        self.outw_x = np.zeros(n, n_input)  # output weights from X
        self.outw_i = np.zeros(n, n_input)  # output weights from previous hidden
        self.outw_c = np.zeros(n, n)  # output weights from current cell state

        self.cellw_x = np.zeros(n, n_input)  # cell state weights from X
        self.cellw_h = np.zeros(n, n_input)  # cell state  weights from previous hidden



