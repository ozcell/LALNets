from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer
from keras.utils.generic_utils import get_from_module
import warnings

Tr = K.theano.tensor.nlinalg.trace
Diag = K.theano.tensor.nlinalg.ExtractDiag()
Range = K.theano.tensor.ptp
Tensordot = K.theano.tensor.tensordot
Scan = K.theano.scan
Arange = K.theano.tensor.arange

class AcolRegularizer(Regularizer):
    """Regularizer for ACOL.

    # Arguments
        c1: Float; affinity factor.
        c2: Float; balance factor.
        c3: Float; coactivity factor.
        c4: Float; L2 regularization factor.
    """

    def __init__(self, c1=0., c2=0., c3=0., c4=0., balance_type=1):
        self.c1 = K.variable(c1)
        self.c2 = K.variable(c2)
        self.c3 = K.variable(c3)
        self.c4 = K.variable(c4)
        self.balance_type = balance_type

    def __call__(self, x):
        regularization = 0
        Z = x
        n = K.shape(Z)[1]

        Z_bar = Z * K.cast(x>0., K.floatx())
        U = K.dot(Z_bar.T, Z_bar)

        if self.balance_type == 1:
            v = Diag(U).reshape((1,n))
        elif self.balance_type == 2:
            v = K.sum(Z_bar, axis=0).reshape((1,n))
        V = K.dot(v.T, v)

        affinity = (K.sum(U) - Tr(U))/((n-1)*Tr(U))
        balance = (K.sum(V) - Tr(V))/((n-1)*Tr(V))
        coactivity = 0. #K.sum(U) - Tr(U)

        if self.c1:
            regularization += self.c1 * affinity
        if self.c2:
            regularization += self.c2 * (1-balance)
        if self.c3:
            regularization += self.c3 * coactivity
        if self.c4:
            regularization += K.sum(self.c4 * K.square(Z))
            #regularization += K.sum(self.c4 * K.square(Z_bar))

        self.affinity = affinity
        self.balance = balance
        self.coactivity = coactivity
        self.reg = regularization

        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c1': K.cast_to_floatx(self.c1.eval()),
                'c2': K.cast_to_floatx(self.c2.eval()),
                'c3': K.cast_to_floatx(self.c3.eval()),
                'c4': K.cast_to_floatx(self.c4.eval())}


class AcolRegularizerNull(Regularizer):
    """Null regularizer for ACOL.

    # Arguments
        c1: Float; affinity factor.
        c2: Float; balance factor.
        c3: Float; coactivity factor.
        c4: Float; L2 regularization factor.
    """

    def __init__(self, c1=0., c2=0., c3=0., c4=0., k=1, balance_type=3):
        self.c1 = K.variable(c1)
        self.c2 = K.variable(c2)
        self.c3 = K.variable(c3)
        self.c4 = K.variable(c4)
        self.k = k
        self.balance_type = balance_type

    def __call__(self, x):
        regularization = 0
        Z = x
        n = K.shape(Z)[1]

        Z_bar = K.reshape(Z * K.cast(Z>0., K.floatx()), (-1, self.k, n//self.k))
        U = Tensordot(Z_bar, Z_bar, axes=[0,0])

        partials, _  = Scan(calculate_partial_affinity_balance,
                       sequences=[Arange(U.shape[1])], non_sequences = [U, self.k, self.balance_type])

        affinity = K.mean(partials[0])
        balance = K.mean(partials[1])
        coactivity = K.mean(partials[1])

        if self.c1:
            regularization += self.c1 * affinity
        if self.c2:
            regularization += self.c2 * (1-balance)
        if self.c3:
            regularization += self.c3 * coactivity
        if self.c4:
            regularization += K.sum(self.c4 * K.square(Z))
            #regularization += K.sum(self.c4 * K.square(Z_bar))

        self.affinity = affinity
        self.balance = balance
        self.coactivity = coactivity
        self.reg = regularization

        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c1': K.cast_to_floatx(self.c1.eval()),
                'c2': K.cast_to_floatx(self.c2.eval()),
                'c3': K.cast_to_floatx(self.c3.eval()),
                'c4': K.cast_to_floatx(self.c4.eval())}


def identity_hvstacked(shape, scale=1, name=None, dim_ordering='th'):
    scale = shape[1]/float(shape[0])
    a = np.identity(int(1/scale))
    for i in range(1, shape[1]):
        a = np.concatenate((a, np.identity(int(1/scale))),axis=1)
    b = np.copy(a)
    for i in range(1, shape[1]):
        b = np.concatenate((b, a),axis=0)
    return K.variable(b, name=name)

def calculate_partial_affinity_balance(i, U, k, balance_type):
    U_partial = U[:,i,:,i]
    if balance_type == 3:
        v = Diag(U_partial).reshape((1,k))
    elif balance_type == 4:
        v = K.sum(U_partial, axis=0).reshape((1,k))
    V = K.dot(v.T, v)
    affinity = (K.sum(U_partial) - Tr(U_partial))/((k-1)*Tr(U_partial))
    balance = (K.sum(V) - Tr(V))/((k-1)*Tr(V))
    return affinity, balance

# Aliases.

def activity_acol(c1=1., c2=1., c3=0., c4=0.000001, balance_type=1):
    return AcolRegularizer(c1=c1, c2=c2, c3=c3, c4=c4, balance_type=balance_type)

def activity_acol_null(c1=1., c2=1., c3=0., c4=0.000001, k=1, balance_type=3):
    return AcolRegularizerNull(c1=c1, c2=c2, c3=c3, c4=c4, k=k, balance_type=balance_type)

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
