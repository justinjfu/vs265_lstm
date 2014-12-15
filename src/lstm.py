import numpy as np
from objective import Objective, Weights
from activations import Logistic
from collections import namedtuple

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
    def __init__(self, layers):
        """
        :param layers: A list of integers denoting # cells in each layer.
            Ex. [10, 30, 10]
        :param act_f: Activation function on gates
        :param act_g: Activation function on inputs
        :param act_h: Activation function on outputs
        """
        self.layers = layers

    def forward_across_time(self, inputs):
        """
        Run forward pass on the input, computing network outputs at each timestep
        :param input: A N x IN_DIM x T input. IN_DIM is the input dimension of the network
        :return: An N x OUT_DIM x T output. OUT_DIM is the output dimension of the network
        """
        current_in = inputs
        for layer in self.layers:
            current_in = layer.forward_across_time(current_in)
        return current_in

    def gradient(self):
        pass

    def update_layer_weights(self, d_weights):
        for i in range(len(self.layers)):
            self.layers[i].update_layer_weights(d_weights[i])

    def eval_objective(self, trainingIn, trainingOut):
        outputs = self.forward_across_time(trainingIn)
        error = 0
        for i in range(len(trainingIn)):
            diff = np.array(outputs[i]) - trainingOut[i].reshape(outputs[i].shape)
            error += np.linalg.norm(diff)
        return error, outputs

    def numerical_gradient(self, d_weights, trainingIn, trainingOut, perturb_amount = 1e-5):
        #import pdb; pdb.set_trace()
        for i in range(len(self.layers)):
            lstm = self.layers[i]
            weights = lstm.to_weights_array()
            dweights = d_weights[i]
            for wi in range(len(weights)):  # need to iterate through each weight independently
                weight = weights[wi]
                dweight = dweights[wi]  # dweight is where we store the gradients for each particular weight
                for index, val in np.ndenumerate(weight):  # a 'single' weight is a matrix, so we iterate within as well
                    weight[index] -= perturb_amount  # w_ij - e
                    lessObj, _ = self.eval_objective(trainingIn, trainingOut)

                    weight[index] += 2 * perturb_amount  # w_ij + e
                    moreObj, _ = self.eval_objective(trainingIn, trainingOut)
                    diff = moreObj - lessObj  # difference
                    # import pdb; pdb.set_trace()
                    grad = -1 * diff / (2 * perturb_amount)  # negative gradient
                    weight[index] -= perturb_amount  # set weight back to normal
                    dweight[index] = grad  # save gradient, will update all the weights at the end


ForwardIntermediate = namedtuple("ForwardIntermediate",
    "input_a input_b forget_a forget_b a_t_c new_cell_states output_a output_b new_hidden output")

BackIntermediate = namedtuple("BackIntermediate",
    "hidden_deriv output_gate_delta cell_deriv cell_delta forget_delta input_delta")

class LSTMLayerWeights(object):
    """
    Stores all of the weights for a single LSTM layer and computes derivatives given forward and backward
    context.
    """

    def __init__(self, n, n_input, n_output, act_f, act_g, act_h):
        self.n = n  # number of units on this layer
        self.n_input = n_input  # number of inputs into this layer
        self.n_output = n_output

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        self.forgetw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # forget weights from X
        self.forgetw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                           (n, n))  # forget weights from previous hidden

        self.inw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # input weights from X
        self.inw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                       (n, n))  # input weights from previous hidden

        self.outw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # output weights from X
        self.outw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                        (n, n))  # output weights from previous hidden

        self.cellw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (n, n_input))  # cell state weights from X
        self.cellw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (n, n))  # cell state weights from previous hidden

        self.final_output_weights = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                                      (n_output, n))  # layer output weights

    def forward_across_time(self, inputs):
        all_outputs = []
        for n in range(len(inputs)):  # Loop through training examples
            T, D, _ = inputs[n].shape
            shapedInput = inputs[n]
            cs, hs = np.zeros((self.n, 1)), np.zeros((self.n, 1))
            outputs = np.zeros((T, self.n_output, 1))
            for t in range(T):
                intermed = self.forward(cs, hs, shapedInput[t,:])
                cs = intermed.new_cell_states
                hs = intermed.new_hidden
                output_t = intermed.output
                outputs[t] = output_t
            all_outputs.append(outputs)
        return all_outputs

    def forward(self, previous_cell, previous_hidden, previous_layer_input):
        """
        Compute forward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return: [new cell states, new hidden states, output]. All are N-dimensional vectors
        """
        # Compute input gate
        input_a = self.inw_x.dot(previous_layer_input) + self.inw_h.dot(previous_hidden)
        input_b = self.act_f(input_a)  # Input gate outputs

        # Compute forget gate
        forget_a = self.forgetw_x.dot(previous_layer_input) + self.forgetw_h.dot(previous_hidden)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = self.cellw_x.dot(previous_layer_input) + self.cellw_h.dot(previous_hidden)
        new_cell_states = input_b * self.act_g(a_t_c) + forget_b * previous_cell

        # Compute output gates
        output_a = self.outw_x.dot(previous_layer_input) + self.outw_h.dot(previous_hidden)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        new_hidden = output_b * self.act_h(new_cell_states)

        # Compute layer outputs
        output = self.final_output_weights.dot(new_hidden)

        return ForwardIntermediate(input_a, input_b, forget_a, forget_b, a_t_c, new_cell_states, output_a,
                                   output_b, new_hidden, output)

    def backward(self, next_backward_intermediate, current_forward_intermediate, next_forward_intermediate, prev_cell_state):
        """
        Compute backward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return:
        """
        forward = current_forward_intermediate
        next_forward = next_forward_intermediate
        next_backward = next_backward_intermediate

        # Hidden State TODO(Justin): Figure out this
        hidden_deriv = self.final_output_weights.dot(next_hidden_delta)  # +????

        # Output gate
        output_gate_delta = self.act_f.deriv(forward.output_a) \
                            * np.sum(np.outer(self.act_h(forward.new_cell_states), hidden_deriv))

        # Cell States
        cell_deriv = forward.output_b * self.act_h.deriv(forward.new_cell_states) * hidden_deriv + \
                     next_forward.forget_b * next_backward.cell_deriv + \
                     self.inw_c.dot(next_backward.input_delta) + \
                     self.forgetw_c.dot(next_backward.forget_delta) + \
                     self.outw_c.dot(next_backward.output_gate_delta)
        cell_delta = forward.input_b * self.act_g.deriv(forward.a_t_c) * cell_deriv

        # Forget gate
        forget_delta = self.act_f.deriv(forward.forget_a) * np.sum(np.outer(prev_cell_state, cell_deriv))

        # Input gate
        input_delta = self.act_f.deriv(forward.input_a) * np.sum(np.outer(self.act_g(forward.a_t_c), cell_deriv))

        return BackIntermediate(hidden_deriv, output_gate_delta, cell_deriv, cell_delta, forget_delta, input_delta)

    def gradient(self, inputs):
        all_outputs = []
        for n in range(len(inputs)):  # Loop through training examples

            T, D, _ = inputs[n].shape
            shapedInput = inputs[n]


            cs, hs = np.zeros((self.n, 1)), np.zeros((self.n, 1))
            forward_intermediates = [None]*T
            for t in range(T):
                intermed = self.forward(cs, hs, shapedInput[t,:])
                cs = intermed.new_cell_states
                hs = intermed.new_hidden
                forward_intermediates[t] = intermed
            #all_outputs.append(outputs)

            backward_intermediates = [None]*T
            future_backward_intermediate = None
            previous_cell_state = forward_intermediates[T-2].new_cell_states
            for t in range(T)[::-1]:
                intermed = self.backward(future_backward_intermediate,
                                         forward_intermediates[t],
                                         previous_cell_state)
                future_backward_intermediate = intermed
                if t > 0:
                    previous_cell_state = forward_intermediates[t-1].new_cell_states
                else:
                    previous_cell_state = np.zeros((self.n, 1))
                backward_intermediates[t] = intermed

            # Calculate gradient
            # TODO.
        return all_outputs

    def update_layer_weights(self, dweights):
        self.forgetw_x += dweights[0]
        self.forgetw_h += dweights[1]

        self.inw_x += dweights[2]
        self.inw_h += dweights[3]

        self.outw_x += dweights[4]
        self.outw_h += dweights[5]

        self.cellw_x += dweights[6]
        self.cellw_h += dweights[7]
        self.final_output_weights += dweights[8]


    def to_weights_array(self):
        return [self.forgetw_x,
                self.forgetw_h,

                self.inw_x,
                self.inw_h,

                self.outw_x,
                self.outw_h,

                self.cellw_x,
                self.cellw_h,
                self.final_output_weights]

    
if __name__ == '__main__':
    # test on bitstring parity checker - tests feed forward only with numerical gradient calculation
    N = 1
    trainingIn1 = np.array([[1, 1, 1, 1, 1, 1, 0, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8*N,2,1)
    trainingIn2 = np.array([[1, 0, 1, 0] * N, [0,0,0,0]*N]).T.reshape(4*N,2,1)
    trainingIn3 = np.array([[0, 0, 0, 1, 0, 1, 1, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8*N,2,1)
    trainingIn4 = np.array([[1, 0, 1, 1, 1, 1] * N, [0,0,0,0,0,0]*N]).T.reshape(6*N,2,1)
    trainingIn5 = np.array([[0, 1, 0, 0, 1, 0, 1] * N, [0,0,0,0,0,0,0]*N]).T.reshape(7*N,2,1)

    trainingIn = [trainingIn1, trainingIn2, trainingIn3, trainingIn4, trainingIn5]
    trainingOut1 = (np.cumsum(trainingIn1, axis=0) % 2)[:,0,:]
    trainingOut2 = (np.cumsum(trainingIn2, axis=0) % 2)[:,0,:]
    trainingOut3 = (np.cumsum(trainingIn3, axis=0) % 2)[:,0,:]
    trainingOut4 = (np.cumsum(trainingIn4, axis=0) % 2)[:,0,:]
    trainingOut5 = (np.cumsum(trainingIn5, axis=0) % 2)[:,0,:]

    trainingOut = [trainingOut1, trainingOut2, trainingOut3, trainingOut4, trainingOut5]

    f, g, h = Logistic(), Logistic(), Logistic()
    lstm_layer1 = LSTMLayerWeights(2, 2, 1, f, g, h)
    #lstm_layer2 = LSTMLayerWeights(1, 2, f, g, h)
    d_weight1 = [np.zeros(w.shape) for w in lstm_layer1.to_weights_array()]
    #d_weight2 = [np.zeros(w.shape) for w in lstm_layer2.to_weights_array()]

    d_weights = [d_weight1]

    lstm = LSTMNetwork([lstm_layer1])

    for trial in range(500):
        lstm.numerical_gradient(d_weights, trainingIn, trainingOut, perturb_amount = 1e-5)
        lstm.update_layer_weights(d_weights)
        if (trial+1) % 100 == 0:
            err, output = lstm.eval_objective(trainingIn, trainingOut)
            print "Trial =", trial+1
            print err
            print output

    print "TESTING INPUT NAO"
    testIn1 = np.array([[1, 0, 1] * N, [0,0,0]*N]).T.reshape(3,2,1)
    testIn2 = np.array([[1, 0, 0, 1] * N, [0,0,0,0]*N]).T.reshape(4,2,1)
    testIn3 = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1] * N, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]*N]).T.reshape(23,2,1)
    testIn = [testIn1, testIn2, testIn3]
    testOut1 = (np.cumsum(testIn1, axis=0) % 2)[:,0,:]
    testOut2 = (np.cumsum(testIn2, axis=0) % 2)[:,0,:]
    testOut3 = (np.cumsum(testIn3, axis=0) % 2)[:,0,:]

    testOut = [testOut1, testOut2, testOut3]
    obj, output = lstm.eval_objective(testIn, testOut)
    print testOut
    print output
    print obj


