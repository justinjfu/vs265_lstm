from lstm import *
import numpy as np
from descend import gd
import argparse, pickle, socket, traceback 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('trainingIn1')
    parser.add_argument('trainingIn2')
    args = parser.parse_args()
   
    with open(args.trainingIn1, 'rb') as f:
        trainingIn1 = pickle.load(f)
    with open(args.trainingIn2, 'rb') as f:
        trainingIn2 = pickle.load(f)
    gravity = np.array([0,9.8])
    trainingIn1_shaped = [np.array([point[1:] - gravity for point in blah]).reshape(len(blah),2,1) for blah in trainingIn1]
    trainingIn2_shaped = [np.array([point[1:] - gravity for point in blah]).reshape(len(blah),2,1) for blah in trainingIn2]
    #trainingOut1_shaped = [np.array([[1,0] if i > len(blah)/2 else [0,0] for i in range(len(blah))]) for blah in trainingIn1]
    #trainingOut2_shaped = [np.array([[0,1] if i > len(blah)/2 else [0,0] for i in range(len(blah))]) for blah in trainingIn2]
    trainingOut1_shaped = [np.array([[1] for i in range(len(blah))]) for blah in trainingIn1]
    trainingOut2_shaped = [np.array([[0] for i in range(len(blah))]) for blah in trainingIn2]

    trainingIn = trainingIn1_shaped + trainingIn2_shaped
    trainingOut = trainingOut1_shaped + trainingOut2_shaped


    f, g, h = Logistic(), Logistic(), Tanh()
    lstm_layer1 = LSTMLayerWeights(2, 2, 1, f, g, h)
    d_weight1 = [np.zeros(w.shape) for w in lstm_layer1.to_weights_array()]

    d_weights = [d_weight1]

    lstm = LSTMNetwork([lstm_layer1])

    for trial in range(1000):
        #lstm.numerical_gradient(d_weights, trainingIn, trainingOut)
        d_weights = lstm.gradient(trainingIn, trainingOut)
        d_weights = [[weight*0.05 for weight in layer] for layer in d_weights]
        lstm.update_layer_weights(d_weights)
        if (trial+1) % 100 == 0:
            err, output = lstm.eval_objective(trainingIn, trainingOut)
            print "Trial =", trial+1
            print err
            print output

    print "FINAL WEIGHTS"
    final_weights = lstm.layers[0].to_weights_array()
    for final_wt in final_weights:
        print final_wt
        print ""
    with open('gestures.lstm', 'wb') as datfile:
        pickle.dump(lstm, datfile)

