import gzip
import cPickle
from sklearn.preprocessing import OneHotEncoder
from theano_utils import NP_FLOATX
import numpy as np

def get_mnist():
    print '... loading data'
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    print '... done'
    return train_set, valid_set, test_set


def preprocess(dataset, twoD=False):
    data, labels = dataset
    N, dims = data.shape

    if twoD:
        data = data.reshape(N, 1, 28, 28)
    #import pdb; pdb.set_trace()

    enc = OneHotEncoder(n_values=10)
    train_labels = labels.reshape(N,1)
    enc.fit(train_labels)
    labels = enc.transform(train_labels).toarray().astype(NP_FLOATX)
    return data, labels

def error(predict, gold):
    pred_idx = np.argmax(predict, axis=1)
    gold_idx = np.argmax(gold, axis=1)
    err = np.sum(pred_idx != gold_idx)
    N, d = gold.shape
    print err, "/", N
    return float(err)/N