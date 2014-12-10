import numpy as np
from objective import Objective, Weights


class LSTMObjective(Objective):
    def __init__(self, training_set):
        super(LSTMObjective, self).__init__()
        self.weights = None
        self.training = training_set

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


class LSTMLayerWeights(object):
    def __init__(self, n, n_input, act_f, act_g, act_h):
        self.n = n  # number of units on this layer
        self.n_input = n_input  # number of inputs into this layer

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        self.forgetw_x = np.zeros(n, n_input)  # forget weights from X
        self.forgetw_h = np.zeros(n, n)  # forget weights from previous hidden
        self.forgetw_c = np.zeros(n, n)  # forget weights from previous cell state

        self.inw_x = np.zeros(n, n_input)  # input weights from X
        self.inw_h = np.zeros(n, n)  # input weights from previous hidden
        self.inw_c = np.zeros(n, n)  # input weights from previous cell state

        self.outw_x = np.zeros(n, n_input)  # output weights from X
        self.outw_h = np.zeros(n, n)  # output weights from previous hidden
        self.outw_c = np.zeros(n, n)  # output weights from current cell state

        self.cellw_x = np.zeros(n, n_input)  # cell state weights from X
        self.cellw_h = np.zeros(n, n)  # cell state weights from previous hidden

        self.final_output_weights = np.zeros(n, n) # layer output weights

    def forward(self, previous_cell, previous_hidden, previous_layer_input):
        """
        Compute forward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return: [new cell states, new hidden states, output]. All are N-dimensional vectors
        """
        # Compute input gate
        input_a = self.inw_x.dot(previous_layer_input) + self.inw_h.dot(previous_hidden) + self.inw_c.dot(previous_cell)
        input_b = self.act_f(input_a)  # Input gate outputs

        # Compute forget gate
        forget_a = self.forgetw_x.dot(previous_layer_input) + self.forgetw_h.dot(previous_hidden) + self.forgetw_c.dot(previous_cell)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = self.cellw_x.dot(previous_layer_input) + self.cellw_h.dot(previous_hidden)
        new_cell_states = input_b*self.act_g(a_t_c) + forget_b*previous_cell

        # Compute output gates
        output_a = self.outw_x.dot(previous_layer_input) + self.outw_h.dot(previous_hidden) + self.outw_c.dot(new_cell_states)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        new_hidden = output_b*self.act_h(new_cell_states)

        # Compute layer outputs
        output = self.final_output_weights.dot(new_hidden)

        return new_cell_states, new_hidden, output

    def backward(self, next_cell, next_hidden, next_layer_output):
        """
        Compute backward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return:
        """
        pass