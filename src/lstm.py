import numpy as np
from objective import Objective, Weights
from activations import Logistic

WEIGHT_INIT_RANGE = 0.1


class LSTMObjective(Objective):
    def __init__(self, training_set):
        super(LSTMObjective, self).__init__()
        self.weights = None
        self.training = training_set

    def gradient_at(self, wts):
        raise NotImplemented

    def value_at(self, wts, label):
        return 0.5 * (wts - label) ** 2


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


class LSTMNetwork(object):
    def __init__(self, layers, act_f, act_g, act_h):
        """
        :param layers: A list of integers denoting # cells in each layer.
            Ex. [10, 30, 10]
        :param act_f: Activation function on gates
        :param act_g: Activation function on inputs
        :param act_h: Activation function on outputs
        """
        pass

    def forward_eval(self, inputs):
        """
        Run forward pass on the input, computing network outputs at each timestep
        :param input: A N x IN_DIM x T input. IN_DIM is the input dimension of the network
        :return: An N x OUT_DIM x T output. OUT_DIM is the output dimension of the network
        """
        pass

    def gradient(self):
        pass


class LSTMLayerWeights(object):
    """
    Stores all of the weights for a single LSTM layer and computes derivatives given forward and backward
    context.
    """

    def __init__(self, n, n_input, n_output, act_f, act_g, act_h):
        self.n = n  # number of units on this layer
        self.n_input = n_input  # number of inputs into this layer

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        self.forgetw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # forget weights from X
        self.forgetw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                           (n, n))  # forget weights from previous hidden
        self.forgetw_c = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                           (n, n))  # forget weights from previous cell state

        self.inw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # input weights from X
        self.inw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                       (n, n))  # input weights from previous hidden
        self.inw_c = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                       (n, n))  # input weights from previous cell state

        self.outw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # output weights from X
        self.outw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                        (n, n))  # output weights from previous hidden
        self.outw_c = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                        (n, n))  # output weights from current cell state

        self.cellw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (n, n_input))  # cell state weights from X
        self.cellw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (n, n))  # cell state weights from previous hidden

        self.final_output_weights = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                                      (n_output, n))  # layer output weights

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
        forget_a = self.forgetw_x.dot(previous_layer_input) + self.forgetw_h.dot(previous_hidden) + self.forgetw_c.dot(
            previous_cell)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = self.cellw_x.dot(previous_layer_input) + self.cellw_h.dot(previous_hidden)
        new_cell_states = input_b * self.act_g(a_t_c) + forget_b * previous_cell

        # Compute output gates
        output_a = self.outw_x.dot(previous_layer_input) + self.outw_h.dot(previous_hidden) + self.outw_c.dot(
            new_cell_states)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        new_hidden = output_b * self.act_h(new_cell_states)

        # Compute layer outputs
        output = self.final_output_weights.dot(new_hidden)

        return new_cell_states, new_hidden, output

    def backward(self):
        """
        Compute backward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return:
        """
        # TODO(Justin): Thread information from forward pass into this function, and debug

        # Hidden State TODO(Justin): Figure out this
        hidden_delta = self.final_output_weights.dot(next_hidden_delta)  # +????

        # Output gate
        output_gate_delta = self.act_f.deriv(output_a) * np.sum(np.outer(self.act_h(new_cell_states), hidden_delta))

        # Cell States
        cell_deriv = output_b * self.act_h.deriv(new_cell_states) * hidden_delta + \
                     next_forget_b * next_cell_delta + \
                     self.inw_c.dot(next_input_delta) + \
                     self.forgetw_c.dot(next_forget_delta) + \
                     self.outw_c.dot(next_output_gate_delta)
        cell_delta = input_b * self.act_g.deriv(a_t_c) * cell_deriv

        # Forget gate
        forget_delta = self.act_f.deriv(forget_a) * np.sum(np.outer(previous_cell, cell_deriv))

        # Input gate
        input_delta = self.act_f.deriv(input_a) * np.sum(np.outer(self.act_g(a_t_c), cell_deriv))

    def update_layer_weights(self, dweights):
        self.forgetw_x += dweights[0]
        self.forgetw_h += dweights[1]
        self.forgetw_c += dweights[2]

        self.inw_x += dweights[3]
        self.inw_h += dweights[4]
        self.inw_c += dweights[5]

        self.outw_x += dweights[6]
        self.outw_h += dweights[7]
        self.outw_c += dweights[8]

        self.cellw_x += dweights[9]
        self.cellw_h += dweights[10]
        self.final_output_weights += dweights[11]


    def to_weights_array(lstm):
        return np.array([lstm.forgetw_x,
                        lstm.forgetw_h,
                        lstm.forgetw_c,

                        lstm.inw_x,
                        lstm.inw_h,
                        lstm.inw_c,

                        lstm.outw_x,
                        lstm.outw_h,
                        lstm.outw_c,

                        lstm.cellw_x,
                        lstm.cellw_h,
                        lstm.final_output_weights])

if __name__ == '__main__':
    # test on bitstring parity checker - tests feed forward only with numerical gradient calculation
    f, g, h = Logistic(), Logistic(), Logistic()
    lstm = LSTMLayerWeights(2, 1, 1, f, g, h)

    N = 1
    trainingIn = np.array([1, 0, 0, 0, 0, 0, 0, 1] * N)
    trainingOut = np.cumsum(trainingIn) % 2

    weights = lstm.to_weights_array()

    d_weights = np.array([np.zeros(lstm.forgetw_x.shape),
                          np.zeros(lstm.forgetw_h.shape),
                          np.zeros(lstm.forgetw_c.shape),

                          np.zeros(lstm.inw_x.shape),
                          np.zeros(lstm.inw_h.shape),
                          np.zeros(lstm.inw_c.shape),

                          np.zeros(lstm.outw_x.shape),
                          np.zeros(lstm.outw_h.shape),
                          np.zeros(lstm.outw_c.shape),

                          np.zeros(lstm.cellw_x.shape),
                          np.zeros(lstm.cellw_h.shape),
                          np.zeros(lstm.final_output_weights.shape)])

    def eval_objective(lstm_object, trainingIn, trainingOut):
        #import pdb; pdb.set_trace()
        outputs = np.zeros(trainingOut.shape)
        cs, hs = np.zeros((2, 1)), np.zeros((2, 1))
        for t in range(len(trainingIn)):
            # _, _, temp_out_old = lstm.forward(cs, hs, trainingIn[t]) # the output for this training sample
            #lessObj = ObjF.value_at(temp_out_old, trainingOut[t]) # the error with negative perturb
            cs, hs, output = lstm_object.forward(cs, hs, trainingIn[t])  # calc the real error
            outputs[t] = output
        diff = np.array(outputs) - trainingOut
        return np.linalg.norm(diff), outputs

    perturb_amount = 1e-5
    ObjF = LSTMObjective(trainingIn)

    print eval_objective(lstm, trainingIn, trainingOut)
    for trial in range(300):
        #import pdb; pdb.set_trace()
        for wi in range(len(weights)):  # need to iterate through each weight independently
            weight = weights[wi]
            dweight = d_weights[wi]  # dweight is where we store the gradients for each particular weight
            for index, val in np.ndenumerate(weight):  # a 'single' weight is a matrix, so we iterate within as well
                weight[index] -= perturb_amount  # w_ij - e
                lessObj, _ = eval_objective(lstm, trainingIn, trainingOut)

                weight[index] += 2 * perturb_amount  # w_ij + e
                moreObj, _ = eval_objective(lstm, trainingIn, trainingOut)
                diff = moreObj - lessObj  # difference
                # import pdb; pdb.set_trace()
                grad = -1 * diff / (2 * perturb_amount)  # negative gradient
                weight[index] -= perturb_amount  # set weight back to normal
                dweight[index] = grad  # save gradient, will update all the weights at the end
        #print 'D_weights:', d_weights
        #print 'weights:', weights
        lstm.update_layer_weights(0.5*d_weights)
        obj, output = eval_objective(lstm, trainingIn, trainingOut)
        print obj
        print output
    print trainingOut


