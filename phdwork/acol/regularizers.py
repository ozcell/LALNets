from __future__ import absolute_import
from keras import backend as K
from keras.regularizers import Regularizer
from phdwork.acol.initializations import column_vstacked
from keras.utils.generic_utils import get_from_module
import warnings

Tr = K.theano.tensor.nlinalg.trace
Diag = K.theano.tensor.nlinalg.ExtractDiag()
Range = K.theano.tensor.ptp

class AcolRegularizer(Regularizer):
    """Regularizer for ACOL.

    # Arguments
        c1: Float; affinity factor.
        c2: Float; balance factor.
        c3: Float; coactivity factor.
        c4: Float; L2 regularization factor.
    """

    def __init__(self, c1=0., c2=0., c3=0., c4=0.):
        self.c1 = K.variable(c1)
        self.c2 = K.variable(c2)
        self.c3 = K.variable(c3)
        self.c4 = K.variable(c4)

    def __call__(self, x):
        regularization = 0
        Z = x
        n = K.shape(Z)[1]

        Z_bar = Z * K.cast(x>0., K.floatx())
        #v = K.sum(Z_bar, axis=0).reshape((1,n))

        U = K.dot(Z_bar.T, Z_bar)
        v = Diag(U).reshape((1,n))
        V = K.dot(v.T, v)

        affinity = (K.sum(U) - Tr(U))/((n-1)*Tr(U))
        balance = (K.sum(V) - Tr(V))/((n-1)*Tr(V))
        coactivity = K.sum(U) - Tr(U)

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

    def __init__(self, c1=0., c2=0., c3=0., c4=0., k=1):
        self.c1 = K.variable(c1)
        self.c2 = K.variable(c2)
        self.c3 = K.variable(c3)
        self.c4 = K.variable(c4)
        self.k = k

    def __call__(self, x):
        regularization = 0
        Z = x
        n = K.cast_to_floatx(K.int_shape(Z)[1])

        Z_bar = Z * K.cast(Z>0., K.floatx())

        mask = column_vstacked((n, self.k))
        n = self.k
        Z_bar = K.dot(Z_bar, mask)

        U = K.dot(Z_bar.T, Z_bar)
        v = Diag(U).reshape((1,n))
        V = K.dot(v.T, v)

        affinity = (K.sum(U) - Tr(U))/((n-1)*Tr(U))
        balance = (K.sum(V) - Tr(V))/((n-1)*Tr(V))
        coactivity = K.sum(U) - Tr(U)

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


# Aliases.

def activity_acol(c1=1., c2=1., c3=0., c4=0.000001,):
    return AcolRegularizer(c1=c1, c2=c2, c3=c3, c4=c4)

def activity_acol_null(c1=1., c2=1., c3=0., c4=0.000001, k=1):
    return AcolRegularizerNull(c1=c1, c2=c2, c3=c3, c4=c4, k=k)

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
