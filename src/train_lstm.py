from lstm import *
from activations import *
import numpy as np
from descend import gd
import argparse, pickle, socket, traceback 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('trainingIn1')
    parser.add_argument('trainingIn2')
    parser.add_argument('trainingIn3')

    args = parser.parse_args()
   
    with open(args.trainingIn1, 'rb') as f:
        trainingIn1 = pickle.load(f)
    with open(args.trainingIn2, 'rb') as f:
        trainingIn2 = pickle.load(f)
    with open(args.trainingIn3, 'rb') as f:
        trainingIn3 = pickle.load(f)

    #gravity = np.array([0,9.8])
    trainingIn1_shaped = [np.array([point[1:] for point in blah]).reshape(len(blah),2,1) for blah in trainingIn1]
    trainingIn2_shaped = [np.array([point[1:] for point in blah]).reshape(len(blah),2,1) for blah in trainingIn2]
    trainingIn3_shaped = [np.array([point[1:] for point in blah]).reshape(len(blah),2,1) for blah in trainingIn3]

    #trainingOut1_shaped = [np.array([[1,0] if i > len(blah)/2 else [0,0] for i in range(len(blah))]) for blah in trainingIn1]
    #trainingOut2_shaped = [np.array([[0,1] if i > len(blah)/2 else [0,0] for i in range(len(blah))]) for blah in trainingIn2]
    trainingOut1_shaped = [np.array([[1, 0] for i in range(len(blah))]) for blah in trainingIn1_shaped]
    trainingOut2_shaped = [np.array([[0, 1] for i in range(len(blah))]) for blah in trainingIn2_shaped]
    trainingOut3_shaped = [np.array([[0, 0,1] for i in range(len(blah))]) for blah in trainingIn3_shaped]

    trainingIn = trainingIn1_shaped + trainingIn2_shaped #+ trainingIn3_shaped
    trainingOut = trainingOut1_shaped + trainingOut2_shaped #+ trainingOut3_shaped


    f, g, h = Logistic(), Logistic(), Tanh()
    lstm_layer1 = LSTMLayerWeights(2, 2, 2, f, g, h)
    lstm_layer2 = LSTMLayerWeights(5, 5, 3, f, g, h)
    lstm_layer3 = LSTMLayerWeights(4, 6, 1, f, g, h)


    lstm = LSTMNetwork([lstm_layer1])

    d_weight1 = [np.zeros(w.shape) for w in lstm_layer1.to_weights_array()]
    d_weights = [d_weight1]

    """
    for trial in range(100):
        #import pdb; pdb.set_trace()
        #lstm.numerical_gradient(d_weights, trainingIn, trainingOut, perturb_amount = 1e-5)
        d_weights = lstm.gradient(trainingIn, trainingOut)
        lstm.update_layer_weights(d_weights, K=-0.001)
        if (trial+1) % 1 == 0:
            err, output = lstm.eval_objective(trainingIn, trainingOut)
            print "Trial =", trial+1
            print err
            #print output
            print Softmax.softmax(output[-1][-1])
            print trainingOut[-1][-1]
    """

    wt = LSTMWeights(lstm.to_weights_array())
    obj = LSTMObjective(trainingIn, trainingOut, lstm)
    wt = gd(obj, wt, iters=300, heartbeat=5, learning_rate = 0.001, momentum_rate = 0.1)

    print "FINAL WEIGHTS"
    final_weights = lstm.layers[0].to_weights_array()
    for final_wt in final_weights:
        print final_wt
        print ""
    with open('gestures.lstm', 'wb') as datfile:
        pickle.dump(lstm, datfile)

