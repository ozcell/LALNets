from __future__ import absolute_import, division
import numpy as np
from keras import backend as K
from keras.utils.generic_utils import get_from_module

def identity_vstacked(shape, scale=1, name=None, dim_ordering='th'):
    scale = shape[1]/float(shape[0])
    a = np.identity(shape[1])
    for i in range(1, int(1/scale)):
        a = np.concatenate((a, np.identity(shape[1])),axis=0)
    return K.variable(a, name=name)

def column_vstacked(shape, scale=1, name=None, dim_ordering='th'):
    scale = shape[1]/float(shape[0])
    b = np.zeros((1,shape[1]))
    b[0,0] = 1
    a = np.copy(b)
    for i in range(1, int(1/scale)):
        a = np.concatenate((a, b),axis=0)
    for j in range(1, shape[1]):
        b = np.zeros((1,shape[1]))
        b[0,j] = 1
        for i in range(0, int(1/scale)):
            a = np.concatenate((a, b),axis=0)
    return K.variable(a, name=name)

def column_vstacked_nullnode(shape, scale=1, name=None, dim_ordering='th'):
    scale = (shape[1]-1)/float(shape[0])
    b = np.zeros((1,shape[1]))
    b[0,0] = 1
    a = np.copy(b)
    for i in range(1, int(1/scale)):
        a = np.concatenate((a, b),axis=0)
    for j in range(1, shape[1]-1):
        b = np.zeros((1,shape[1]))
        b[0,j] = 1
        for i in range(0, int(1/scale)):
            a = np.concatenate((a, b),axis=0)
    return K.variable(a, name=name)

def identity_dstacked(shape, scale=1, name=None, dim_ordering='th'):
    scale = shape[1]/float(shape[0])
    a = np.identity(shape[1])
    for i in range(1, int(1/scale)):
        a = np.concatenate((a, np.identity(shape[1])),axis=0)

    b = np.expand_dims(np.diag(a[:,0]), axis=2)
    for i in range(1, shape[1]):
        c = np.expand_dims(np.diag(a[:,i]), axis=2)
        b = np.concatenate((b, c),axis=2)
    return K.variable(b, name=name)

def get(identifier, **kwargs):
    return get_from_module(identifier, globals(),
                           'initialization', kwargs=kwargs)
