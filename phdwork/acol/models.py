'''

Model generator for ACOL experiments.

'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm

from phdwork.acol.layers.pooling import AveragePooling, MaxPooling
from phdwork.acol.regularizers import activity_acol


def define_cnn(input_shape, nb_classes, cnn_type=1, conv_params=(32,3,2), hidden_drop = True,
               acol_params=(5, 0, 1, 1, 0, 0.000001, 'average', False),
               init='identity_vstacked', null_node= False, truncated = False):

    nb_filters, nb_conv, nb_pool = conv_params

    K, p, c1, c2, c3, c4, pooling, trainable = acol_params

    if pooling == 'average':
        AcolPooling = AveragePooling
    elif pooling == 'max':
        AcolPooling = MaxPooling

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th',
                            input_shape=input_shape))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='th'))
    model.add(Dropout(0.2)) if hidden_drop else model.add(Dropout(0.))

    if cnn_type>1:
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv,
                               activation='relu', border_mode='same', dim_ordering='th'))
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv,
                               activation='relu', border_mode='same', dim_ordering='th'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='th'))
        model.add(Dropout(0.3)) if hidden_drop else model.add(Dropout(0.))

    if cnn_type==3:
        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th'))
        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='th'))
        model.add(Dropout(0.4)) if hidden_drop else model.add(Dropout(0.))

    if cnn_type>3:

        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th'))
        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th'))
        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='th'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='th'))
        model.add(Dropout(0.4)) if hidden_drop else model.add(Dropout(0.))

    model.add(Flatten())
    if cnn_type < 5:
        model.add(Dense(2048, activation='relu'))
    else:
        model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5)) if hidden_drop else model.add(Dropout(0.))

    model.add(Dense(nb_classes*K, activity_regularizer=activity_acol(c1, c2, c3, c4), name='L-1'))
    model.add(Dropout(p))

    if not truncated:
        model.add(Activation('softmax', name='L-1_activation'))

        if null_node:
            model.add(AcolPooling(nb_classes+1, trainable=trainable, init='column_vstacked_nullnode', name='AcolPooling'))
        else:
            model.add(AcolPooling(nb_classes, trainable=trainable, init=init, name='AcolPooling'))

    return model


def define_mlp(input_shape, nb_classes, mlp_params=(3, 2048, 0., 0.5, 2.),
               acol_params=(5, 0, 1, 1, 0, 0.000001, 'average', False),
               init='identity_vstacked', null_node= False, truncated = False):


    nb_layers, nb_nodes, p_i, p_hl, m_n = mlp_params
    K, p, c1, c2, c3, c4, pooling, trainable = acol_params

    if pooling == 'average':
        AcolPooling = AveragePooling
    elif pooling == 'max':
        AcolPooling = MaxPooling

    model = Sequential()
    model.add(Dropout(p_i, input_shape=input_shape))
    for layer in range(nb_layers):
        model.add(Dense(nb_nodes, activation='relu', W_constraint=maxnorm(m_n)))
        model.add(Dropout(p_hl))

    model.add(Dense(nb_classes*K, activity_regularizer=activity_acol(c1, c2, c3, c4), name='L-1'))
    model.add(Dropout(p))

    if not truncated:
        model.add(Activation('softmax', name='L-1_activation'))
        if null_node:
            model.add(AcolPooling(nb_classes+1, trainable=trainable, init='column_vstacked_nullnode', name='AcolPooling'))
        else:
            model.add(AcolPooling(nb_classes, trainable=trainable, init=init, name='AcolPooling'))

    return model
