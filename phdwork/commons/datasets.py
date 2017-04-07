from keras.datasets import mnist

def load_mnist(order='tf'):

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

    return (X_train, y_train), (X_test, y_test)
