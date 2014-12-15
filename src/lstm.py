import numpy as np
from objective import Objective, Weights
from activations import Logistic, Tanh
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
        all_outputs = []
        for i in range(len(inputs)):
            current_in = inputs[i]
            for layer in self.layers:
                intermediates = layer.forward_across_time(current_in)
                current_in = np.array([intermed.output for intermed in intermediates])
            all_outputs.append(current_in)
        return all_outputs

    def gradient(self, inputs, outputs):
        Nlayers = len(self.layers)
        layer_intermediates = [None]*Nlayers
        gradients = []
        for layer in self.layers:
            weights_in_layer = [np.zeros(weight.shape) for weight in layer.to_weights_array()]
            gradients.append(weights_in_layer)
        for i in range(len(inputs)):  # loop over training examples
            current_in = inputs[i]
            for j in range(len(self.layers)):
                intermediates = self.layers[j].forward_across_time(current_in)
                layer_intermediates[j] = intermediates
            final_layer_output = np.array([intermed.output for intermed in layer_intermediates[Nlayers-1]])

            next_layer_del_k = self.output_backprop_error(final_layer_output, outputs[i])
            for j in range(len(self.layers))[::-1]:
                gradient, next_layer_del_k = self.layers[j].gradient(layer_intermediates[j], next_layer_del_k)
                for k in range(len(gradients[j])):
                    gradients[j][k] += gradient[k]
        return gradients

    def update_layer_weights(self, d_weights):
        for i in range(len(self.layers)):
            self.layers[i].update_layer_weights(d_weights[i])

    def eval_objective(self, trainingIn, trainingOut):
        outputs = self.forward_across_time(trainingIn)
        error = 0
        for i in range(len(trainingIn)):
            diff = np.array(outputs[i]) - trainingOut[i].reshape(outputs[i].shape)
            diff = diff.reshape(len(diff))
            error += 0.5*np.dot(diff,diff)
        return error*100, outputs

    def output_backprop_error(self, output, trainingOut):
        return -(trainingOut.reshape(output.shape) - np.array(output))*100

    def numerical_gradient(self, d_weights, trainingIn, trainingOut, perturb_amount = 1e-5):
        #import pdb; pdb.set_trace()
        #original_objective, _ = self.eval_objective(trainingIn, trainingOut)
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
                    #lessObj = original_objective

                    weight[index] += 2* perturb_amount  # w_ij + e
                    moreObj, _ = self.eval_objective(trainingIn, trainingOut)
                    diff = moreObj - lessObj  # difference
                    # import pdb; pdb.set_trace()
                    grad = -1 * diff / (2* perturb_amount)  # negative gradient
                    weight[index] -= perturb_amount  # set weight back to normal
                    dweight[index] = grad  # save gradient, will update all the weights at the end


ForwardIntermediate = namedtuple("ForwardIntermediate",
    "previous_layer_input input_a input_b forget_a forget_b a_t_c new_cell_states output_a output_b new_hidden output_pre output")

ForwardIntermediateForgetB = namedtuple("ForwardIntermediateForgetB", "forget_b")


BackIntermediate = namedtuple("BackIntermediate",
    "del_k_pre hidden_deriv output_gate_delta cell_deriv cell_delta forget_delta input_delta del_k del_h")

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

        self.init_back_intermed = self.init_dels()

    
    def init_dels(self):
        del_k_pre = np.zeros((self.n_output,1))
        hidden_deriv = np.zeros((self.n,1))

        # Output gate
        output_gate_delta = np.zeros((self.n,1))

        # Cell States
        cell_deriv = np.zeros((self.n,1))
        cell_delta = np.zeros((self.n,1))

        # Forget gate
        forget_delta = np.zeros((self.n,1))

        # Input gate
        input_delta = np.zeros((self.n,1))

        del_k = np.zeros((self.n,1))

        del_h = np.zeros((self.n,1))
        return BackIntermediate(del_k_pre,
                                hidden_deriv,
                                output_gate_delta,
                                cell_deriv,
                                cell_delta,
                                forget_delta,
                                input_delta,
                                del_k,
                                del_h)

    def forward_across_time(self, input):
        T, D, _ = input.shape
        shapedInput = input
        cs, hs = np.zeros((self.n, 1)), np.zeros((self.n, 1))
        intermediates = []
        for t in range(T):
            intermed = self.forward(cs, hs, shapedInput[t,:])
            cs = intermed.new_cell_states
            hs = intermed.new_hidden
            intermediates.append(intermed)
        return intermediates

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
        output_pre = self.final_output_weights.dot(new_hidden)
        output = self.act_h(output_pre)

        return ForwardIntermediate(previous_layer_input, input_a, input_b, forget_a, forget_b, a_t_c, new_cell_states, output_a,
                                   output_b, new_hidden, output_pre, output)

    def backward(self, next_backward, forward, next_forward, prev_cell_state, next_layer_del_k):
        """
        Compute backward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return:
        """
        # Hidden State
        del_k_pre = next_layer_del_k*self.act_h.deriv(forward.output_pre)

        #import pdb; pdb.set_trace()
        del_h = self.inw_h.dot(next_backward.input_delta) + \
                self.forgetw_h.dot(next_backward.forget_delta) +\
                self.outw_h.dot(next_backward.output_gate_delta) + \
                self.cellw_h.T.dot(next_backward.cell_delta)

        hidden_deriv = self.final_output_weights.T.dot(del_k_pre) + del_h

        # Output gate
        output_gate_delta = self.act_f.deriv(forward.output_a) \
                            * self.act_h(forward.new_cell_states) * hidden_deriv

        #import pdb; pdb.set_trace()
        # Cell States
        cell_deriv = forward.output_b * self.act_h.deriv(forward.new_cell_states) * hidden_deriv + \
                     next_forward.forget_b * next_backward.cell_deriv
        cell_delta = forward.input_b * self.act_g.deriv(forward.a_t_c) * cell_deriv

        # Forget gate
        forget_delta = self.act_f.deriv(forward.forget_a) * prev_cell_state * cell_deriv

        # Input gate
        input_delta = self.act_f.deriv(forward.input_a) * self.act_g(forward.a_t_c) * cell_deriv

        del_k = self.inw_x.T.dot(input_delta) + \
                self.forgetw_x.T.dot(forget_delta) + \
                self.outw_x.T.dot(output_gate_delta) + \
                self.cellw_x.T.dot(cell_delta)
                #cell_delta * np.sum(self.cellw_x, axis=0, keepdims=True)

        return BackIntermediate(del_k_pre, hidden_deriv, output_gate_delta, cell_deriv, cell_delta, forget_delta, input_delta,
                                del_k, del_h)

    def gradient(self, forward_intermediates, next_layer_del_k):
        """
        Gradient for 1 input
        :param forward_list:
        :param next_layer_del_k:
        :return:
        """
        #import pdb; pdb.set_trace()
        T = len(forward_intermediates)
        backward_intermediates = [None]*T
        future_backward_intermediate = self.init_back_intermed
        for t in range(T)[::-1]:
            if t == T-1:
                next_forward = ForwardIntermediateForgetB(np.zeros((self.n, 1)))
            else:
                next_forward = forward_intermediates[t+1]

            if t == 0:
                previous_cell_state = np.zeros((self.n, 1))
            else:
                previous_cell_state = forward_intermediates[t-1].new_cell_states

            intermed = self.backward(future_backward_intermediate,
                                     forward_intermediates[t],
                                     next_forward,
                                     previous_cell_state,
                                     next_layer_del_k[t])
            future_backward_intermediate = intermed

            backward_intermediates[t] = intermed

        # Calculate gradient
        forgetw_x_g = np.zeros((self.n, self.n_input))
        inw_x_g = np.zeros((self.n, self.n_input))
        outw_x_g = np.zeros((self.n, self.n_input))
        cellw_x_g = np.zeros((self.n, self.n_input))

        forgetw_h_g = np.zeros((self.n, self.n))
        inw_h_g = np.zeros((self.n, self.n))
        outw_h_g = np.zeros((self.n, self.n))
        cellw_h_g = np.zeros((self.n, self.n))

        final_output_g = np.zeros((self.n_output, self.n))

        for t in range(T)[::-1]:
            b_i = backward_intermediates[t]
            f_i = forward_intermediates[t]
            forgetw_x_g += np.outer(b_i.forget_delta, f_i.previous_layer_input)
            inw_x_g += np.outer(b_i.input_delta, f_i.previous_layer_input)
            outw_x_g += np.outer(b_i.output_gate_delta, f_i.previous_layer_input)
            cellw_x_g += np.outer(b_i.cell_delta, f_i.previous_layer_input)

            if t == 0:
                f_i_past_hidden = np.zeros((self.n, 1))
            else:
                f_i_past_hidden = forward_intermediates[t-1].new_hidden

            forgetw_h_g += np.outer(b_i.forget_delta, f_i_past_hidden)
            inw_h_g += np.outer(b_i.input_delta, f_i_past_hidden)
            outw_h_g += np.outer(b_i.output_gate_delta, f_i_past_hidden)
            cellw_h_g += np.outer(b_i.cell_delta, f_i_past_hidden)

            final_output_g += np.outer(f_i.new_hidden, b_i.del_k_pre).T
        
        gradient = [-forgetw_x_g,
                    -forgetw_h_g,

                    -inw_x_g,
                    -inw_h_g,

                    -outw_x_g,
                    -outw_h_g,

                    -cellw_x_g,
                    -cellw_h_g,
                                  
                    -final_output_g]

        layer_del_k = [x.del_k for x in backward_intermediates]

        return gradient, layer_del_k

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
    np.random.seed(20)
    # test on bitstring parity checker - tests feed forward only with numerical gradient calculation
    N = 1
    trainingIn0 = np.array([[1, 0] * N, [0, 0]*N]).T.reshape(2*N,2,1)
    trainingIn1 = np.array([[1, 1, 1, 1, 1, 1, 0, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8*N,2,1)
    trainingIn2 = np.array([[1, 0, 1, 0] * N, [0,0,0,0]*N]).T.reshape(4*N,2,1)
    trainingIn3 = np.array([[0, 0, 0, 1, 0, 1, 1, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8*N,2,1)
    trainingIn4 = np.array([[1, 0, 1, 1, 1, 1] * N, [0,0,0,0,0,0]*N]).T.reshape(6*N,2,1)
    trainingIn5 = np.array([[0, 1, 0, 0, 1, 0, 1] * N, [0,0,0,0,0,0,0]*N]).T.reshape(7*N,2,1)

    #trainingIn = [trainingIn1, trainingIn2, trainingIn3, trainingIn4, trainingIn5]
    trainingIn = [trainingIn0]
    trainingOut0 = (np.cumsum(trainingIn0, axis=0) % 2)[:,0,:]
    trainingOut1 = (np.cumsum(trainingIn1, axis=0) % 2)[:,0,:]
    trainingOut2 = (np.cumsum(trainingIn2, axis=0) % 2)[:,0,:]
    trainingOut3 = (np.cumsum(trainingIn3, axis=0) % 2)[:,0,:]
    trainingOut4 = (np.cumsum(trainingIn4, axis=0) % 2)[:,0,:]
    trainingOut5 = (np.cumsum(trainingIn5, axis=0) % 2)[:,0,:]

    #trainingOut = [trainingOut1, trainingOut2, trainingOut3, trainingOut4, trainingOut5]
    trainingOut = [trainingOut0]

    f, g, h = Logistic(), Logistic(), Tanh()
    lstm_layer1 = LSTMLayerWeights(3, 2, 1, f, g, h)
    #lstm_layer2 = LSTMLayerWeights(1, 2, 1, f, g, h)
    d_weight1 = [np.zeros(w.shape) for w in lstm_layer1.to_weights_array()]
    #d_weight2 = [np.zeros(w.shape) for w in lstm_layer2.to_weights_array()]

    d_weights = [d_weight1]

    lstm = LSTMNetwork([lstm_layer1])

    for trial in range(300):
        #import pdb; pdb.set_trace()
        lstm.numerical_gradient(d_weights, trainingIn, trainingOut, perturb_amount = 1e-6)
        gradient = lstm.gradient(trainingIn, trainingOut)
        lstm.update_layer_weights(d_weights)
        if (trial+1) % 50 == 0:
            err, output = lstm.eval_objective(trainingIn, trainingOut)
            print "Trial =", trial+1
            print err
            print output

    print "TESTING INPUT NAO"
    testIn1 = np.array([[1, 0, 1] * N, [0,0,0]*N]).T.reshape(3,2,1)
    testIn2 = np.array([[1, 0, 0, 1] * N, [0,0,0,0]*N]).T.reshape(4,2,1)
    testIn3 = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1] * N, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]*N]).T.reshape(23,2,1)
    testIn = [testIn1, testIn2, testIn3]
    #testIn = [testIn1, testIn2]
    testOut1 = (np.cumsum(testIn1, axis=0) % 2)[:,0,:]
    testOut2 = (np.cumsum(testIn2, axis=0) % 2)[:,0,:]
    testOut3 = (np.cumsum(testIn3, axis=0) % 2)[:,0,:]

    testOut = [testOut1, testOut2, testOut3]
    #testOut = [testOut1, testOut2]
    obj, output = lstm.eval_objective(testIn, testOut)
    print testOut
    print output
    print obj


