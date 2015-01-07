from lstm import *
import numpy as np
from client_test import collect_data
import argparse, pickle
from activations import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('LSTMNetworks')
    parser.add_argument('class_names')
    args = parser.parse_args()


    networkfiless = args.LSTMNetworks.split(',')
    classes = args.class_names.split(',')

    networks = []
    for netfile in networkfiless:
        with open(netfile, 'rb') as f:
            network = pickle.load(f)
            networks.append(network)
    #print lstm.to_weights_array()

    while True:
        usr_in = raw_input("Press Enter to Start Recording A Sample and Test it, 'q' to Exit Program")
        if usr_in == "":
            trainingIn = collect_data()

            trainingIn_shaped = [np.array([point[1:] for point in trainingIn]).reshape(len(trainingIn),2,1)]

            sumval = 0
            for lstm in networks:
                output = lstm.forward_across_time(trainingIn_shaped)
                netout = Softmax.softmax(output[0])
                netout = np.sum(netout, axis=0)
                sumval = netout + sumval
            sumval = sumval/np.sum(sumval)
            for i in range(len(classes)):
                print '%s : %f'% (classes[i], sumval[i])
            #print Softmax.softmax(sumval)

            #print Softmax.softmax(output[0])
            #print output[0]
            argmax = np.argmax(sumval)
            print "RECOGNIZED:", classes[ argmax]
            sumval[argmax] = 0
            print "SECOND GUESS:",classes[np.argmax(sumval)]
        elif usr_in.lower() == "q":
            break





