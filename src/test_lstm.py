from lstm import *
import numpy as np
from client_test import collect_data
import argparse, pickle
from activations import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('LSTMNetwork')
    parser.add_argument('class_names')
    args = parser.parse_args()
    
    classes = args.class_names.split(',')
    THRESH = 0.5
   
    with open(args.LSTMNetwork, 'rb') as f:
        lstm = pickle.load(f)
    print lstm.to_weights_array()

    while True:
        usr_in = raw_input("Press Enter to Start Recording A Sample and Test it, 'q' to Exit Program")
        if usr_in == "":
            trainingIn = collect_data()

            trainingIn_shaped = [np.array([point[1:] for point in trainingIn]).reshape(len(trainingIn),2,1)]
            output = lstm.forward_across_time(trainingIn_shaped)
            #val = np.sum(output[-10:])/10
            sumval = np.sum(output[0], axis=0)
            sumval = Softmax.softmax(sumval)
            for i in range(len(classes)):
                print '%s : %f'% (classes[i], sumval[i])
            #print Softmax.softmax(sumval)

            #print Softmax.softmax(output[0])
            #print output[0]
            print "RECOGNIZED:", classes[ np.argmax(sumval)]
        elif usr_in.lower() == "q":
            break





