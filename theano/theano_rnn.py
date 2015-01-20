import theano
import theano.tensor as T
from theano_utils import randn, NP_FLOATX
from theano_nn import SquaredLoss, TanhLayer, SigmLayer, SoftMaxLayer
import numpy as np
import theano.typed_list

__author__ = 'justin'


class BaseLayer(object):
    n_instances = 0

    def __init__(self):
        self.layer_id = BaseLayer.n_instances
        BaseLayer.n_instances += 1

    def forward_time(self, prev_layer, prev_state, prev_output):
        """
        Run a forward pass for a single layer and single time step
        :param prev_layer:
        :param prev_state:
        :return: (next_layer, next_state)
        """
        raise NotImplementedError

    def initial_state(self):
        raise NotImplementedError

    def initial_output(self):
        raise NotImplementedError

    def params(self):
        """ Return a list of trainable parameters """
        return []


class FeedForwardLayer(BaseLayer):
    """
    A special adapter for feedforward networks
    """
    def __init__(self):
        super(FeedForwardLayer, self).__init__()
        self.zero = theano.shared(np.array([0.0]))

    def forward(self, prev_layer):
        raise NotImplementedError

    def forward_time(self, prev_layer, prev_state, prev_output):
        return self.forward(prev_layer), self.zero

    def initial_state(self):
        return self.zero  # Unused

    def initial_output(self):
        return self.zero  # Unused

    def params(self):
        raise NotImplementedError


class ActivationLayer(FeedForwardLayer):
    def __init__(self, act):
        super(ActivationLayer, self).__init__()
        self.act = act

    def forward(self, prev_layer):
        return self.act(prev_layer)

    def params(self):
        return []


class FFIPLayer(FeedForwardLayer):
    """ Feedforward inner product layer """
    def __init__(self, n_in, n_out):
        super(FFIPLayer, self).__init__()
        self.w = theano.shared(randn(n_in, n_out), name="ff_ip_w_"+str(self.layer_id))
        self.b = theano.shared(randn(n_out), name="b_ip"+str(self.layer_id))

    def forward(self, prev_layer):
        return prev_layer.dot(self.w) + self.b

    def params(self):
        """ Return a list of trainable parameters """
        return [self.w, self.b]


class RNNIPLayer(BaseLayer):
    """ Recurrent inner product layer """
    def __init__(self, n_in, n_out, act):
        super(RNNIPLayer, self).__init__()
        self.n_out = n_out
        self.w_ff = theano.shared(0.2*randn(n_in, n_out), name="rnn_ip_wff_"+str(self.layer_id))
        self.w_r = theano.shared(0.2*randn(n_out, n_out), name="rnn_ip_wr_"+str(self.layer_id))
        self.bias = theano.shared(0.2*randn(n_out), name="rnn_ip_bias_"+str(self.layer_id))
        self.act = act

    def forward_time(self, prev_layer, _, prev_output):
        new_state = self.w_r.dot(prev_output)
        output = self.act(prev_layer.dot(self.w_ff)+new_state+self.bias)
        return output, output+0

    def initial_output(self):
        return theano.shared(np.zeros(self.n_out))

    def initial_state(self):
        #return np.zeros((self.n_out, 1))  <---- This produces very different numbers
        return theano.shared(np.zeros(self.n_out))

    def params(self):
        """ Return a list of trainable parameters """
        return [self.w_ff, self.w_r]


class LSTMLayer(BaseLayer):
    def __init__(self, n_input, n_out, act_f, act_g, act_h):
        super(LSTMLayer, self).__init__()

        self.n_input = n_input
        self.n_out = n_out

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        def init_weights(d1, d2, name):
            return theano.shared(np.random.uniform(-0.1, 0.1, (d1, d2)), name=name+"_"+str(self.layer_id))

        self.forgetw_x = init_weights(n_out, n_input, "forgetw_x")  # forget weights from X
        self.forgetw_h = init_weights(n_out, n_out, "forgetw_h")  # forget weights from previous hidden

        self.inw_x = init_weights(n_out, n_input, "inw_x")  # input weights from X
        self.inw_h = init_weights(n_out, n_out, "inw_h") # input weights from previous hidden

        self.outw_x = init_weights(n_out, n_input, "outw_x")  # output weights from X
        self.outw_h = init_weights(n_out, n_out, "outw_h")  # output weights from previous hidden

        self.cellw_x = init_weights(n_out, n_input, "cellw_x") # cell state weights from X
        self.cellw_h = init_weights(n_out, n_out, "cellw_h")  # cell state weights from previous hidden

    def forward_time(self, previous_layer, previous_cell_state, previous_output):
        # Compute input gate
        input_a = self.inw_x.dot(previous_layer) + self.inw_h.dot(previous_output)
        input_b = self.act_f(input_a)  # Input gate outputs

        # Compute forget gate
        forget_a = self.forgetw_x.dot(previous_layer) + self.forgetw_h.dot(previous_output)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = self.cellw_x.dot(previous_layer) + self.cellw_h.dot(previous_output)
        new_cell_states = input_b * self.act_g(a_t_c) + forget_b * previous_cell_state

        # Compute output gates
        output_a = self.outw_x.dot(previous_layer) + self.outw_h.dot(previous_output)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        output = output_b * self.act_h(new_cell_states)
        return output, new_cell_states

    def initial_state(self):
        #return np.zeros((self.n_out, 1))  <---- This produces very different numbers
        return theano.shared(np.zeros(self.n_out))

    def initial_output(self):
        return theano.shared(np.zeros(self.n_out))

    def params(self):
        """ Return a list of trainable parameters """
        return [self.forgetw_x, self.forgetw_h, self.inw_x, self.inw_h, self.outw_x, self.outw_h,
                self.cellw_x, self.cellw_h]

class RecurrentNetwork(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        #setup for gradients
        self.params_list = []
        for layer in layers:
            self.params_list.extend(layer.params())

    def params(self):
        return self.params_list

    def predict(self):
        x = T.matrix('X')
        pred = theano.function([x], self.forward_across_time(x))
        return pred

    def forward_across_time(self, training_ex):
        previous_layer = training_ex
        for layer in self.layers:
            hidden_state = layer.initial_state()
            init_state = layer.initial_output()
            def loop(prev_layer, prev_output, prev_state):
                nlayer, nstate = layer.forward_time(prev_layer, prev_state, prev_output)
                return nlayer, nstate

            results, updates = theano.scan(fn=loop,
                                            sequences=[previous_layer],
                                            outputs_info=[dict(initial=init_state, taps=[-1]), hidden_state])
            next_layer, hidden_states = results
            previous_layer = next_layer

        return previous_layer

    def predict_list(self):
        results, results_list, updates, data_list = self.forward_across_time_list()
        pred = theano.function(inputs=[data_list, results_list], outputs=results, updates=updates)
        return pred

    def forward_across_time_list(self):
        raise NotImplementedError  # Lists don't work well yet
        data_list = theano.typed_list.TypedListType(T.dmatrix)(name='data_list')
        length = theano.typed_list.length(data_list)

        results_list = theano.typed_list.TypedListType(T.dmatrix)(name='results_list')
        def ns(): pass
        ns.results_list = results_list

        def loop(i, data_list_arg):
            training_ex = data_list_arg[i]
            output_layer = self.forward_across_time(training_ex)
            ns.results_list = ns.results_list.append(output_layer)
            ns.results_list = ns.results_list.append(training_ex)
            l =  theano.typed_list.length(ns.results_list)
            return i, l
        results, updates = theano.scan(fn=loop,
                    sequences=[T.arange(length, dtype='int64')],
                    outputs_info=[None, None],
                    non_sequences=[data_list])
        return results, ns.results_list, updates, data_list

    def prepare_objective_list(self):
        data_list = theano.typed_list.TypedListType(T.dmatrix)(name='data_list')
        label_list = theano.typed_list.TypedListType(T.dmatrix)(name='label_list')
        length = theano.typed_list.length(data_list)

        def loop(i, data_list_arg, label_list_arg):
            training_ex = data_list_arg[i]
            output_layer = self.forward_across_time(training_ex)
            layer_loss = self.loss.loss(data_list_arg[i], output_layer)
            return layer_loss
        results, updates = theano.scan(fn=loop,
                    sequences=[T.arange(length, dtype='int64')],
                    outputs_info=[None],
                    non_sequences=[data_list, label_list])
        results = T.sum(results)
        return results, data_list, label_list

    def prepare_objective(self, data, labels):
        # data is a list of matrices
        obj = None
        for i in range(len(data)):
            training_ex = data[i]
            label = theano.shared(labels[i])
            net_output = self.forward_across_time(theano.shared(training_ex))
            layer_loss = self.loss.loss(label, net_output)
            if obj:
                obj += layer_loss
            else:
                obj = layer_loss
        return obj

    def prepare_objective_var(self, training_ex, label):
        # data is a list of matrices
        net_output = self.forward_across_time(training_ex)
        layer_loss = self.loss.loss(label, net_output)
        obj = layer_loss
        return obj

import theano.ifelse
def train_gd(obj, params, args, batch_size=10, eta=0.01):
    gradients = T.grad(obj, params)

    ii = theano.shared(0)
    total_grad = [theano.shared(np.zeros_like(param.get_value())) for param in params]

    updates = []

    for i in range(len(params)):
        updates.append((params[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                        params[i]-eta*total_grad[i],
                                                        params[i] )))
    updates.append((ii, ii+1))

    for i in range(len(params)):
        updates.append((total_grad[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                            T.zeros_like(total_grad[i]),
                                                            total_grad[i]+gradients[i]) ))


    train = theano.function(
        inputs=args,
        outputs=[obj, ii],
        updates=updates
    )
    return train

def train_gd_list(obj, params, data_arg, label_arg, eta=0.01):
    print 'PARAMS:', params
    gradients = T.grad(obj, params)
    updates = [None]*len(gradients)

    for i in range(len(gradients)):
        updates[i] = (params[i], params[i]-eta*gradients[i])

    train = theano.function(
        inputs=[data_arg, label_arg],
        outputs=obj,
        updates=updates)
    return train


def train_gd_host(trainable, data, labels, eta=0.01):
    obj = trainable.prepare_objective(data, labels)
    params = trainable.params()
    print 'PARAMS:', params
    gradients = T.grad(obj, params)
    updates = [None]*len(gradients)

    for i in range(len(gradients)):
        updates[i] = (params[i], params[i]-eta*gradients[i])

    train = theano.function(
        inputs=[],
        updates=updates)
    return train


def generate_parity_data(num, l):
    examples = []
    labels = []
    for i in range(num):
        N = np.random.randint(low=l, high=l+1)

        rand_data = np.random.randint(size=(N, 1), low=0, high=2).astype(np.float32)

        rand_label = np.cumsum(rand_data, axis=0) % 2
        #rand_data = np.hstack((rand_data, rand_label))


        examples.append(rand_data)
        labels.append(rand_label)
    return examples, labels


def test_rnn():
    data, labels = generate_parity_data(1)
    print "DATA:", data[0]
    data = theano.shared(data[0])
    label = theano.shared(labels[0])

    w = theano.shared(randn(2, 1))

    def fff(prev_layer, prev_state):
        #import pdb; pdb.set_trace()
        print prev_state
        print prev_layer
        print w.get_value()
        output = prev_layer.dot(w)+prev_state
        return output, output+0

    results, updates = theano.scan(
        fn=fff,
        outputs_info=[None, theano.shared(np.ones((1)))],
        sequences=[data])

    blah = theano.function(inputs=[], outputs=results)

    a = blah()
    print 'Output:', a

    loss = SquaredLoss().loss(label, results[0])
    blah = theano.function(inputs=[], outputs=loss)
    print blah()

    gd = T.grad(loss, [w])
    blah = theano.function(inputs=[], outputs=gd)
    print 'gradient:', blah()


def test_parity():
    data, labels = generate_parity_data(2, 5)
    print 'data:', data[0].T

    l1 = RNNIPLayer(1, 1, T.tanh)
    #l2 = RNNIPLayer(2, 1, T.nnet.sigmoid)


    rnn = RecurrentNetwork([l1], SquaredLoss())
    p = rnn.predict()

    ob, data_arg, label_arg = rnn.prepare_objective_list()
    fff = theano.function([data_arg, label_arg], ob)

    print fff(data, labels)

    train_fn = train_gd_list(ob, rnn.params(), data_arg, label_arg)
    for i in range(100):
        print train_fn(data, labels)
    """
    print p(data[0])
    train_fn = train_gd_host(rnn, data, labels, eta=0.1)

    for i in range(800):
        loss = train_fn()[0]
        if i % 5 == 0:
            print i, ':', loss


    print labels[0]
    p = rnn.predict()
    print p(data[0])

    # Check for generalization
    data, labels = generate_parity_data(2)
    for i in range(len(data)):
        predicted = p(data[0])
        print "New Labels:",labels[0]
        print "New predict:", predicted

    p_list = rnn.predict_list()
    a = []
    print p_list(data, a)
    print a
    """

def test_list():
    #TODO: typed_list doesn't seem to work with append/outputs
    data_list = theano.typed_list.TypedListType(T.dmatrix)(name='data_list')
    length = theano.typed_list.length(data_list)

    out_list = theano.typed_list.TypedListType(T.dmatrix)(name='out_list')
    def ff(i, dat_list, olist):
        theano.typed_list.insert(olist, i, dat_list[i])
        return dat_list[i], theano.typed_list.getitem(olist, i)
    results, updates = theano.scan(
        fn=ff,
        sequences=[T.arange(length, dtype='int64')],
        outputs_info=[None, None],
        non_sequences=[data_list, out_list])

    a = theano.function(inputs=[data_list, out_list], outputs=[results, out_list], updates=updates)

    eye = np.eye
    print a([eye(2), eye(2), eye(2)], [])
    #"""

    """
    out_list = theano.typed_list.TypedListType(T.dscalar)(name='out_list')
    a = T.dscalar('a')
    b = T.dscalar('b')
    res = out_list.append(a).append(b)
    f = theano.function(inputs=[a,b, out_list], outputs=[res])

    print f(2,3, [])
    """



if __name__ == "__main__":
    np.random.seed(10)

    """
    x = T.scalar('x')
    k = T.scalar('k')

    def f(i, xx):
        return xx+k, k

    results, updates = theano.scan(
        fn=f,
        outputs_info=[np.array(0.0), None],
        sequences=T.arange(5))
    blah = theano.function(inputs=[k], outputs=results, updates=updates)
    print blah(1)
    """

    #test_parity()

    data, labels = generate_parity_data(20, 10)
    print 'data:', data[0].T

    #l1 = RNNIPLayer(1, 10, T.tanh)
    #l2 = RNNIPLayer(10, 1, T.nnet.sigmoid)
    l1 = LSTMLayer(1, 2, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh)
    l2 = FFIPLayer(2, 1)
    l3 = ActivationLayer(T.nnet.sigmoid)

    rnn = RecurrentNetwork([l1, l2, l3], SquaredLoss())
    p = rnn.predict()

    data_var = T.matrix('data')
    label_var = T.matrix('labels')
    ob = rnn.prepare_objective_var(data_var, label_var)
    loss_func = theano.function([data_var, label_var], ob)

    print p(data[0])
    train_fn = train_gd(ob, rnn.params(), [data_var, label_var], batch_size=20, eta=0.05)

    for i in range(2000):
        for j in range(len(data)):
            train_fn(data[j], labels[j])

        if i % 20 == 0:
            loss_tot = 0
            for j in range(len(data)):
                loss_tot += loss_func(data[j], labels[j])
            print i, ':', loss_tot


    print labels[0]
    p = rnn.predict()
    print p(data[0])

    data, labels = generate_parity_data(1, 25)
    print labels[0]
    print p(data[0])

    data, labels = generate_parity_data(1, 25)
    print labels[0]
    print p(data[0])


