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

ForwardIntermediate = namedtuple("ForwardIntermediate",
    "input_a input_b forget_a forget_b a_t_c new_cell_states output_a output_b new_hidden output")

BackIntermediate = namedtuple("BackIntermediate",
    "hidden_deriv output_gate_delta cell_deriv cell_delta forget_delta input_delta")

class LSTMLayerWeights(object):
    """
    Stores all of the weights for a single LSTM layer and computes derivatives given forward and backward
    context.
    """

    def __init__(self, n, n_input, act_f, act_g, act_h, num_cells=2):
        self.n = n  # number of units on this layer
        self.n_input = n_input  # number of inputs into this layer

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs
        self.num_cells = num_cells

        self.forgetw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # forget weights from X
        self.forgetw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                           (n, num_cells))  # forget weights from previous hidden
        self.forgetw_c = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                           (n, num_cells))  # forget weights from previous cell state

        self.inw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # input weights from X
        self.inw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                       (n, num_cells))  # input weights from previous hidden
        self.inw_c = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                       (n, num_cells))  # input weights from previous cell state

        self.outw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # output weights from X
        self.outw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                        (n, num_cells))  # output weights from previous hidden
        self.outw_c = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                        (n, num_cells))  # output weights from current cell state

        self.cellw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (num_cells, n, n_input))  # cell state weights from X
        self.cellw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (num_cells, n, n))  # cell state weights from previous hidden

        self.final_output_weights = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                                      (n, num_cells))  # layer output weights

    def forward_across_time(self, inputs):
        all_outputs = []
        for n in range(len(inputs)):  # Loop through training examples
            T, D, _ = inputs[n].shape
            shapedInput = inputs[n]
            cs, hs = np.zeros((self.n, self.num_cells)), np.zeros((self.n, self.num_cells))
            outputs = np.zeros((T, self.n, 1))
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
        import pdb; pdb.set_trace()
        input_a = self.inw_x.dot(previous_layer_input) + self.inw_h.dot(previous_hidden) + np.sum(self.inw_c*(previous_cell), axis=1, keepdims=True)
        input_b = self.act_f(input_a)  # Input gate outputs

        # Compute forget gate
        forget_a = self.forgetw_x.dot(previous_layer_input) + self.forgetw_h.dot(previous_hidden) + np.sum(self.forgetw_c*(
            previous_cell), axis=1, keepdims=True)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = self.cellw_x.dot(previous_layer_input)[:,:,0] + self.cellw_h.dot(previous_hidden)[:,:,0]
        new_cell_states = input_b * self.act_g(a_t_c) + forget_b * previous_cell

        # Compute output gates
        output_a = self.outw_x.dot(previous_layer_input) + self.outw_h.dot(previous_hidden) + np.sum(self.outw_c*(
            new_cell_states), axis=1, keepdims=True)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        new_hidden = output_b * self.act_h(new_cell_states)

        # Compute layer outputs
        output = np.sum(self.final_output_weights*(new_hidden), axis=1, keepdims=True)

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
    N = 1
    trainingIn1 = np.array([[1, 1, 1, 1, 1, 1, 0, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8,2,1)
    trainingIn2 = np.array([[1, 0, 1, 0] * N, [0,0,0,0]*N]).T.reshape(4,2,1)
    trainingIn3 = np.array([[0, 0, 0, 1, 0, 1, 1, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8,2,1)
    trainingIn4 = np.array([[1, 0, 1, 1, 1, 1] * N, [0,0,0,0,0,0]*N]).T.reshape(6,2,1)
    trainingIn5 = np.array([[0, 1, 0, 0, 1, 0, 1] * N, [0,0,0,0,0,0,0]*N]).T.reshape(7,2,1)

    trainingIn = [trainingIn1, trainingIn2, trainingIn3, trainingIn4, trainingIn5]
    trainingOut1 = (np.cumsum(trainingIn1, axis=0) % 2)[:,0,:]
    trainingOut2 = (np.cumsum(trainingIn2, axis=0) % 2)[:,0,:]
    trainingOut3 = (np.cumsum(trainingIn3, axis=0) % 2)[:,0,:]
    trainingOut4 = (np.cumsum(trainingIn4, axis=0) % 2)[:,0,:]
    trainingOut5 = (np.cumsum(trainingIn5, axis=0) % 2)[:,0,:]

    trainingOut = [trainingOut1, trainingOut2, trainingOut3, trainingOut4, trainingOut5]

    f, g, h = Logistic(), Logistic(), Logistic()
    lstm = LSTMLayerWeights(2, 1, f, g, h)


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
        outputs = lstm_object.forward_across_time(trainingIn)

        error = 0
        for i in range(len(trainingIn)):
            diff = np.array(outputs[i]) - trainingOut[i].reshape(outputs[i].shape)
            error += np.linalg.norm(diff)
        return error, outputs

    perturb_amount = 1e-5
    ObjF = LSTMObjective(trainingIn)

    #print eval_objective(lstm, trainingIn, trainingOut)
    for trial in range(500):
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
        lstm.update_layer_weights(d_weights)
        obj, output = eval_objective(lstm, trainingIn, trainingOut)
        if (trial+1) % 100 == 0:
            print "Trial =", trial+1
            print obj
            print output
    print trainingOut


    print "TESTING INPUT NAO"
    testIn1 = np.array([[1, 0, 1] * N, [0,0,0]*N]).T.reshape(3,2,1)
    testIn2 = np.array([[1, 0, 0, 1] * N, [0,0,0,0]*N]).T.reshape(4,2,1)
    testIn3 = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1] * N, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]*N]).T.reshape(23,2,1)
    testIn = [testIn1, testIn2, testIn3]
    testOut1 = (np.cumsum(testIn1, axis=0) % 2)[:,0,:]
    testOut2 = (np.cumsum(testIn2, axis=0) % 2)[:,0,:]
    testOut3 = (np.cumsum(testIn3, axis=0) % 2)[:,0,:]

    testOut = [testOut1, testOut2, testOut3]
    obj, output = eval_objective(lstm, testIn, testOut)
    print testOut
    print output
    print obj


