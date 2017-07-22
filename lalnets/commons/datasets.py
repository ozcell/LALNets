from keras.datasets import mnist
from sklearn.datasets import fetch_rcv1
from lalnets.metagenome.preprocessing import *

import numpy as np
import scipy.io as sio
import pandas as pd

def load_mnist(order='th'):

    # input image dimensions
    img_rows, img_cols, img_channels,  = 28, 28, 1

    if order == 'tf':
        input_shape=(img_rows, img_cols, img_channels)
    elif order == 'th':
        input_shape=(img_channels, img_rows, img_cols)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(len(X_train), input_shape[0], input_shape[1], input_shape[2])
    X_test = X_test.reshape(len(X_test), input_shape[0], input_shape[1], input_shape[2])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return (X_train, y_train), (X_test, y_test), input_shape


def load_svhn(order='th', path=None, extra=False):

    # input image dimensions
    img_rows, img_cols, img_channels,  = 32, 32, 3
    nb_classes = 10

    if order == 'tf':
        input_shape=(img_rows, img_cols, img_channels)
    elif order == 'th':
        input_shape=(img_channels, img_rows, img_cols)

    if path is None:
        train_data = sio.loadmat('/home/ozsel/Jupyter/datasets/svhn/train_32x32.mat')
    else:
        train_data = sio.loadmat(path + 'train_32x32.mat')

    # access to the dict
    X_train = train_data['X']
    if order == 'tf':
        X_train = X_train.reshape(img_channels*img_rows*img_cols, X_train.shape[-1]).T
        X_train = X_train.reshape(len(X_train), input_shape[0], input_shape[1], input_shape[2])
    elif order == 'th':
        X_train = X_train.T.swapaxes(2,3)
    X_train = X_train.astype('float32')
    X_train /= 255

    y_train = train_data['y']
    y_train = y_train.reshape(len(y_train))
    y_train = y_train%nb_classes

    del train_data

    if path is None:
        test_data = sio.loadmat('/home/ozsel/Jupyter/datasets/svhn/test_32x32.mat')
    else:
        test_data = sio.loadmat(path + 'test_32x32.mat')

    # access to the dict
    X_test = test_data['X']
    if order == 'tf':
        X_test = X_test.reshape(img_channels*img_rows*img_cols, X_test.shape[-1]).T
        X_test = X_test.reshape(len(X_test), input_shape[0], input_shape[1], input_shape[2])
    elif order == 'th':
        X_test = X_test.T.swapaxes(2,3)
    X_test = X_test.astype('float32')
    X_test /= 255

    y_test= test_data['y']
    y_test = y_test.reshape(y_test.shape[0])
    y_test = y_test%nb_classes

    del test_data

    if extra:
        if path is None:
            extra_data = sio.loadmat('/home/ozsel/Jupyter/datasets/svhn/extra_32x32.mat')
        else:
            extra_data = sio.loadmat(path + 'extra_32x32.mat')

        # access to the dict
        X_extra = extra_data['X']
        if order == 'tf':
            X_extra = X_extra.reshape(img_channels*img_rows*img_cols, X_extra.shape[-1]).T
            X_extra = X_extra.reshape(len(X_extra), input_shape[0], input_shape[1], input_shape[2])
        elif order == 'th':
            X_extra = X_extra.T.swapaxes(2,3)
        X_extra = X_extra.astype('float32')
        X_extra /= 255

        y_extra= extra_data['y']
        y_extra = y_extra.reshape(y_extra.shape[0])
        y_extra = y_extra%nb_classes

        del extra_data
    else:
        X_extra = None
        y_extra = None

    return (X_train, y_train), (X_test, y_test), input_shape, (X_extra, y_extra)


def load_norb(path=None):

    # input image dimensions
    img_rows, img_cols, img_channels,  = 96, 96, 2

    input_shape=(img_channels, img_rows, img_cols)

    if path is None:
        X_train = np.load('/home/ozsel/Jupyter/datasets/norb/X_train.npy')
        y_train = np.load('/home/ozsel/Jupyter/datasets/norb/y_train.npy')
    else:
        X_train = np.load(path + 'X_train.npy')
        y_train = np.load(path + 'y_train.npy')

    X_train = X_train.astype('float32')
    #X_train /= 255

    if path is None:
        X_test = np.load('/home/ozsel/Jupyter/datasets/norb/X_test.npy')
        y_test = np.load('/home/ozsel/Jupyter/datasets/norb/y_test.npy')
    else:
        X_test = np.load(path + 'X_test.npy')
        y_test = np.load(path + 'y_test.npy')

    X_test = X_test.astype('float32')
    #X_test /= 255

    return (X_train, y_train), (X_test, y_test), input_shape


def load_sar11(path=None, label_type='parent', miniseqs_size=2000, nb_parents=100):

    nb_samples = 75

    if label_type == 'pseudo_complete' or label_type == 'pseudo_mini':
        loc = '/home/ozsel/Jupyter/datasets/metagenome/metagenome75'
    elif 'parent':
        loc = '/home/ozsel/Jupyter/datasets/metagenome/metagenome75_sparse'
    df = pd.read_csv(loc, header=0, sep=',')

    if label_type == 'pseudo_complete':
        X = get_pseudo_labels_comlete(df, nb_samples, nb_parents)
    elif label_type == 'pseudo_mini':
        X = get_pseudo_labels_mini(df, nb_samples, miniseqs_size, nb_parents)
    elif label_type == 'parent':
        X = get_parent_labels_wrt_gene_call(df, nb_samples)
        nb_parents=X.shape[0]/nb_samples

    np.random.shuffle(X)

    X_train = X[:,:,[3,4]].reshape(X.shape[0], X.shape[1]*2)
    X_train = X_train.astype('float32')
    X_train = (X_train)/21

    y_train_pseudo = X[:,0,1].astype('int')
    sample_ids = X[:,0,0].astype('int')
    sample_id_map = np.load('/home/ozsel/Jupyter/datasets/metagenome/sample_id_map_shorten')

    input_shape = (X_train.shape[1],)

    return (X_train, y_train_pseudo), (sample_ids, sample_id_map), nb_parents, input_shape


def load_reuters(nb_words=2000, test_split=0.2):

    rcv1 = fetch_rcv1()

    ind_ccat = (rcv1.target[:,33] == 1).toarray().reshape(804414)
    ind_ecat = (rcv1.target[:,59] == 1).toarray().reshape(804414)
    ind_gcat = (rcv1.target[:,70] == 1).toarray().reshape(804414)
    ind_mcat = (rcv1.target[:,102] == 1).toarray().reshape(804414)

    ind_valid = np.logical_or(np.logical_and(np.logical_xor(ind_ccat, ind_mcat), np.logical_and(~ind_gcat, ~ind_ecat)),
                              np.logical_and(np.logical_xor(ind_gcat, ind_ecat), np.logical_and(~ind_ccat, ~ind_mcat)))

    y = rcv1.target[ind_valid,].toarray()[:,[33,59,70,102]].argmax(axis=1)

    ind_word = np.argsort(np.bincount(rcv1.data[ind_valid,].nonzero()[1]))[::-1][0:nb_words]

    X = rcv1.data[ind_valid,][:,ind_word].toarray()

    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = y[:int(len(X) * (1 - test_split))]

    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = y[int(len(X) * (1 - test_split)):]

    input_shape = (nb_words,)

    return (X_train, y_train), (X_test, y_test), input_shape
