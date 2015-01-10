from collections import namedtuple
from objective import Weights
import math_interface as np
__author__ = 'justin'

WEIGHT_INIT_RANGE = 0.1


class LayerBase(object):
    def forward_across_time(self, input):
        raise NotImplementedError

    def gradient(self, forward_intermediates, next_layer_del_k):
        raise NotImplementedError

    def update_layer_weights(self, dweights, K=1):
        self.set_weights(self.weights+(dweights * K))

    def set_weights(self, wts):
        raise NotImplementedError

    def to_compact_weights(self):
        raise NotImplementedError

    def make_zero_weights(self):
        raise NotImplementedError


def BuildCompactNetWeights(arg_names):
    """
    Builds compact network weights, which is a list of matrices
    :param arg_names:
    :return:
    """
    arg_names = arg_names.split(' ')
    numargs = len(arg_names)
    class CompactNetWeights(Weights):
        def __init__(self, *args):
            if len(args) != numargs:
                raise ValueError('Expected %d args. Got %d' % (numargs, len(args)))
            self.weights = args
            for i in range(len(arg_names)):
                setattr(self, arg_names[i], args[i])

        def __iter__(self):
            for wt in self.weights:
                yield wt

        def __add__(self, other_weight):
            #import pdb; pdb.set_trace()
            return CompactNetWeights(*
                [(self.weights[mat] + other_weight.weights[mat]) for mat in range(len(self.weights)) ]
            )

        def mul_scalar(self, other_scalar):
            return CompactNetWeights(*[[mat * other_scalar for mat in layer] for layer in self.weights])

        def dot_weight(self, other):
            total = 0
            for mat in range(len(self.weights)):
                other_mat = other.weights[mat]
                self_mat = self.weights[mat]
                total += np.sum(self_mat*other_mat)
            return total
    return CompactNetWeights


ForwardIntermediate = namedtuple("ForwardIntermediate",
    "previous_layer_input input_a input_b forget_a forget_b a_t_c new_cell_states output_a output_b output")

ForwardIntermediateForgetB = namedtuple("ForwardIntermediateForgetB", "forget_b")

ForwardIntermediateNN = namedtuple("ForwardIntermediateNN", "input output_pre output")

BackIntermediate = namedtuple("BackIntermediate",
    "hidden_deriv output_gate_delta cell_deriv cell_delta forget_delta input_delta del_k del_h")


NNWeights = BuildCompactNetWeights("ip bias")
class NNLayer(LayerBase):
    def __init__(self, n_input, n_output, act, usebias=True):
        self.n_input = n_input  # number of inputs into this layer
        self.n_output = n_output
        self.usebias = usebias

        self.act = act
        ip = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n_output, n_input))
        if self.usebias:
            bias = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n_output, 1))
        else:
            bias = np.random.uniform(0, 0, (n_output, 1))

        self.weights = NNWeights(ip, bias)

    def forward_across_time(self, input):
        T, D, _ = input.shape
        intermediates = []
        for t in range(T):
            intermed = self.forward(input[t, :])
            intermediates.append(intermed)
        return intermediates

    def forward(self, previous_layer_input):
        output_pre = self.weights.ip.dot(previous_layer_input)+self.weights.bias

        return ForwardIntermediateNN(
            input=previous_layer_input,
            output_pre=output_pre,
            output=self.act(output_pre)
        )

    def gradient(self, forward_intermediates, next_layer_del_k):
        """
        Gradient for 1 input
        :param forward_list:
        :param next_layer_del_k:
        :return:
        """
        T = len(forward_intermediates)
        backward_intermediates = [None]*T

        # Calculate gradient
        bias_g = np.zeros((self.n_output, 1))
        final_output_g = np.zeros((self.n_output, self.n_input))

        for t in range(T)[::-1]:
            f_i = forward_intermediates[t]
            d_output_pre = next_layer_del_k[t] * self.act.deriv(f_i.output_pre)
            backward_intermediates[t] = d_output_pre
            if self.usebias:
                bias_g += d_output_pre
            final_output_g += np.outer(f_i.input, d_output_pre).T

        gradient = NNWeights(final_output_g, bias_g)

        layer_del_k = [self.weights.ip.T.dot(x) for x in backward_intermediates]

        return gradient, layer_del_k

    def set_weights(self, wts):
        self.weights = wts

    def to_compact_weights(self):
        return self.weights

    def make_zero_weights(self):
        weights_in_layer = [np.zeros(weight.shape) for weight in self.to_compact_weights()]
        return NNWeights(*weights_in_layer)

LSTMWeights = BuildCompactNetWeights("forgetw_x forgetw_h inw_x inw_h outw_x outw_h cellw_x cellw_h")
class LSTMLayerWeights(LayerBase):
    """
    Stores all of the weights for a single LSTM layer and computes derivatives given forward and backward
    context.
    """

    def __init__(self, n_input, n, act_f, act_g, act_h):
        self.n = n  # number of units on this layer
        self.n_input = n_input  # number of inputs into this layer

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        forgetw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # forget weights from X
        forgetw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                           (n, n))  # forget weights from previous hidden

        inw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # input weights from X
        inw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                       (n, n))  # input weights from previous hidden

        outw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE, (n, n_input))  # output weights from X
        outw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                        (n, n))  # output weights from previous hidden

        cellw_x = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (n, n_input))  # cell state weights from X
        cellw_h = np.random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE,
                                         (n, n))  # cell state weights from previous hidden

        self.init_back_intermed = self.init_dels()

        self.weights = LSTMWeights(
                forgetw_x, forgetw_h,
                inw_x,inw_h,
                outw_x,outw_h,
                cellw_x,cellw_h)


    def init_dels(self):
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
        return BackIntermediate(hidden_deriv,
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
            hs = intermed.output
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
        input_a = self.weights.inw_x.dot(previous_layer_input) + self.weights.inw_h.dot(previous_hidden)
        input_b = self.act_f(input_a)  # Input gate outputs

        # Compute forget gate
        forget_a = self.weights.forgetw_x.dot(previous_layer_input) + self.weights.forgetw_h.dot(previous_hidden)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = self.weights.cellw_x.dot(previous_layer_input) + self.weights.cellw_h.dot(previous_hidden)
        new_cell_states = input_b * self.act_g(a_t_c) + forget_b * previous_cell

        # Compute output gates
        output_a = self.weights.outw_x.dot(previous_layer_input) + self.weights.outw_h.dot(previous_hidden)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        output = output_b * self.act_h(new_cell_states)

        return ForwardIntermediate(previous_layer_input, input_a, input_b, forget_a, forget_b, a_t_c, new_cell_states, output_a,
                                   output_b, output)

    def backward(self, next_backward, forward, next_forward, prev_cell_state, next_layer_del_k):
        """
        Compute backward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return:
        """
        # Hidden State
        del_h = self.weights.inw_h.T.dot(next_backward.input_delta) + \
                self.weights.forgetw_h.T.dot(next_backward.forget_delta) +\
                self.weights.outw_h.T.dot(next_backward.output_gate_delta) +\
                self.weights.cellw_h.T.dot(next_backward.cell_delta)

        hidden_deriv = next_layer_del_k + del_h

        # Output gate
        output_gate_delta = self.act_f.deriv(forward.output_a) \
                            * self.act_h(forward.new_cell_states) * hidden_deriv

        # Cell States
        cell_deriv = forward.output_b * self.act_h.deriv(forward.new_cell_states) * hidden_deriv + \
                     next_forward.forget_b * next_backward.cell_deriv
        cell_delta = forward.input_b * self.act_g.deriv(forward.a_t_c) * cell_deriv

        # Forget gate
        forget_delta = self.act_f.deriv(forward.forget_a) * prev_cell_state * cell_deriv

        # Input gate
        input_delta = self.act_f.deriv(forward.input_a) * self.act_g(forward.a_t_c) * cell_deriv

        del_k = self.weights.inw_x.T.dot(input_delta) + \
                self.weights.forgetw_x.T.dot(forget_delta) + \
                self.weights.outw_x.T.dot(output_gate_delta) + \
                self.weights.cellw_x.T.dot(cell_delta)
                #cell_delta * np.sum(self.cellw_x, axis=0, keepdims=True)

        return BackIntermediate(hidden_deriv, output_gate_delta, cell_deriv, cell_delta, forget_delta, input_delta,
                                del_k, del_h)

    def gradient(self, forward_intermediates, next_layer_del_k):
        """
        Gradient for 1 input
        :param forward_list:
        :param next_layer_del_k:
        :return:
        """
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
                f_i_past_hidden = forward_intermediates[t-1].output

            forgetw_h_g += np.outer(b_i.forget_delta, f_i_past_hidden)
            inw_h_g += np.outer(b_i.input_delta, f_i_past_hidden)
            outw_h_g += np.outer(b_i.output_gate_delta, f_i_past_hidden)
            cellw_h_g += np.outer(b_i.cell_delta, f_i_past_hidden)

        gradient = LSTMWeights(
                    forgetw_x_g,forgetw_h_g,
                    inw_x_g,inw_h_g,
                    outw_x_g,outw_h_g,
                    cellw_x_g,cellw_h_g)

        layer_del_k = [x.del_k for x in backward_intermediates]

        return gradient, layer_del_k

    def set_weights(self, wts):
        self.weights = wts

    def to_compact_weights(self):
        return self.weights

    def make_zero_weights(self):
        weights_in_layer = [np.zeros(weight.shape) for weight in self.to_compact_weights()]
        return LSTMWeights(*weights_in_layer)