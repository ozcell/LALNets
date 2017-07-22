'''

Model generator for robustness experiments.

'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm


def define_cnn(input_shape, nb_classes, cnn_type=1, nb_filters = 32, nb_pool = 2, nb_conv = 3):
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='tf',
                            input_shape=input_shape))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='tf'))
    model.add(Dropout(0.25))

    if cnn_type>1:
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv,
                               activation='relu', border_mode='same', dim_ordering='tf'))
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv,
                               activation='relu', border_mode='same', dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='tf'))
        model.add(Dropout(0.25))

    if cnn_type>2:
        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='tf'))
        model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                            activation='relu', border_mode='same', dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering='tf'))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def define_mlp(input_shape, nb_classes):

    model = Sequential()
    model.add(Dense(2048, activation='relu', W_constraint=maxnorm(2.), input_shape=input_shape))
    model.add(Dropout(0.50))

    model.add(Dense(2048, activation='relu', W_constraint=maxnorm(2.)))
    model.add(Dropout(0.50))

    model.add(Dense(2048, activation='relu', W_constraint=maxnorm(2.)))
    model.add(Dropout(0.50))

    model.add(Dense(nb_classes, activation='relu'))

    return model
