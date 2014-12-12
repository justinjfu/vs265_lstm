import numpy as np
from objective import Objective, Weights
from activations import Logistic


class LSTMObjective(Objective):
    def __init__(self, training_set):
        super(LSTMObjective, self).__init__()
        self.weights = None
        self.training = training_set

    def gradient_at(self, wts):
        raise NotImplemented

    def value_at(self, wts, label):
        return 0.5*(wts - label)**2

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
    def __init__(self, n, n_input, n_output, act_f, act_g, act_h):
        self.n = n  # number of units on this layer
        self.n_input = n_input  # number of inputs into this layer

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        self.forgetw_x = np.random.uniform(-1, 1, (n, n_input))  # forget weights from X
        self.forgetw_h = np.random.uniform(-1, 1, (n, n))  # forget weights from previous hidden
        self.forgetw_c = np.random.uniform(-1, 1, (n, n))  # forget weights from previous cell state

        self.inw_x = np.random.uniform(-1, 1, (n, n_input))  # input weights from X
        self.inw_h = np.random.uniform(-1, 1, (n, n))  # input weights from previous hidden
        self.inw_c = np.random.uniform(-1, 1, (n, n))  # input weights from previous cell state

        self.outw_x = np.random.uniform(-1, 1, (n, n_input))  # output weights from X
        self.outw_h = np.random.uniform(-1, 1, (n, n))  # output weights from previous hidden
        self.outw_c = np.random.uniform(-1, 1, (n, n))  # output weights from current cell state

        self.cellw_x = np.random.uniform(-1, 1, (n, n_input))  # cell state weights from X
        self.cellw_h = np.random.uniform(-1, 1, (n, n))  # cell state weights from previous hidden

        self.final_output_weights = np.random.uniform(-1, 1, (n_output, n)) # layer output weights

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

    def backward(self, nxt_cell, next_hidden, next_layer_output):
        """
        Compute backward activations
        :param previous_cell:
        :param previous_hidden:
        :param X:
        :return:
        """
        pass

if __name__ == '__main__':
    # test on bitstring parity checker - tests feed forward only with numerical gradient calculation
    f, g, h = Logistic(), Logistic(), Logistic() 
    lstm = LSTMLayerWeights(2, 1, 1, f, g, h)
    
    N=10
    trainingIn = np.array([1,0,0,0,0,0,0,1]*N)
    trainingOut = np.cumsum(trainingIn)
   
    weights = np.array([lstm.forgetw_x,
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



    perturb_amount = 1e-5
    ObjF = LSTMObjective(trainingIn) 
    cs, hs = np.zeros((2,1)), np.zeros((2,1))
    for t in range(len(trainingIn)):
        for wi in range(len(weights)): # need to iterate through each weight independently
            weight = weights[wi]
            dweight = d_weights[wi] # dweight is where we store the gradients for each particular weight
            for index, val in np.ndenumerate(weight): # a 'single' weight is a matrix, so we iterate within as well

                weight[index] -= perturb_amount # w_ij - e
                _, _, temp_out_old = lstm.forward(cs, hs, trainingIn[t]) # the output for this training sample 
                lessObj = ObjF.value_at(temp_out_old, trainingOut[t]) # the error with negative perturb

                weight[index] += 2*perturb_amount # w_ij + e
                _, _, temp_out_new = lstm.forward(cs, hs, trainingIn[t]) # output for this training sample
                moreObj = ObjF.value_at(temp_out_new, trainingOut[t]) # the error with positive perturb
                diff = moreObj - lessObj # difference
                #import pdb; pdb.set_trace()
                grad = -1*diff/(2*perturb_amount) # negative gradient
                weight[index] -= perturb_amount # set weight back to normal
                dweight[index] = grad # save gradient, will update all the weights at the end
        weights += d_weights # update all the weights for this training sample!
        cs, hs, o = lstm.forward(cs, hs, trainingIn[t]) # calc the real error
        #import pdb; pdb.set_trace()
        print o, ObjF.value_at(o, trainingOut[t])
            
    
