'''
Reproduces the results of

"Clustering-based Source-aware Assessment of True Robustness for Learning Models"

for tested neural network models. Reproduced results might be slightly different due to
seleceted seed of random generator.

'''

from __future__ import print_function

from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import svm, metrics

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import time

from models import define_cnn, define_mlp

np.random.seed(1337)

# input image dimensions
img_rows, img_cols, img_channels,  = 28, 28, 1
input_shape=(img_rows, img_cols, img_channels)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

nb_clusters = 5
nb_epoch = 150
nb_rerun = 100
batch_size = 128
#set model type
model_type = 'CNN'
cnn_type = 1

#optimizer settings
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)

#select which clustering scheme to use
ACOL_clusters = True
if ACOL_clusters:
    est = np.load('./files/robust_validation_est.npy')
else:
    est = np.load('./files/robust_validation_est_kmeans.npy')

#concatenate existing partition
X_all = np.concatenate((X_train,X_test), axis=0)
Y_all = np.concatenate((Y_train,Y_test), axis=0)
y_all = np.concatenate((y_train,y_test), axis=0)

if model_type is not 'CNN':
    input_shape = (img_rows*img_cols*img_channels,)
    X_all = X_all.reshape(X_all.shape[0],img_rows*img_cols*img_channels)
    Y_all = Y_all.reshape(Y_all.shape[0],nb_classes)

for nb_samples in [100, 300, 1000, 3000, 10000, 30000, 60000]:

    start = time.time()

    acc_x = []
    acc_i = []
    class_acc_x = np.zeros((nb_rerun, nb_classes))
    class_acc_i = np.zeros((nb_rerun, nb_classes))

    val_index_length = 0
    #train for source-exclusive
    for rerun in range(nb_rerun):

        val_index = []
        train_index = []

        # divide dataset accourding to exclusive source-aware partitioning
        for i in range(nb_classes):
            val_cluster = np.random.randint(0, nb_clusters)

            val_select = np.where(est==nb_classes*val_cluster+i)[0]
            val_index.extend(val_select)

            train_select = np.where(np.logical_and(est%nb_classes == i,est/nb_classes != val_cluster))[0]
            if nb_samples == 60000:
                train_select = train_select[np.random.permutation(train_select.shape[0])]
            else:
                train_select = train_select[np.random.permutation(train_select.shape[0])[0:(nb_samples/nb_classes)]]
            train_index.extend(train_select)

        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        val_index_length += len(val_index)

        if model_type == 'CNN':
            model = define_cnn(input_shape, nb_classes, cnn_type)
        elif model_type == 'MLP':
            model = define_mlp(input_shape, nb_classes)
        elif model_type == 'SVM':
            model = svm.SVC(C=1,gamma=0.01)

        if model_type == 'SVM':
            model.fit(X_all[train_index,], y_all[train_index])

            expected = y_all[val_index]
            predicted = model.predict(X_all[val_index,])

            acc_x.append(metrics.accuracy_score(expected, predicted))
            for i in range(nb_classes):
                class_acc_x[rerun,i] = metrics.accuracy_score(expected[y_all[val_index]==i],
                                                              predicted[y_all[val_index]==i])

        else:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

            acc_x.append([[],[],[],[]])
            history = model.fit(X_all[train_index,], Y_all[train_index,],
                                batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                                validation_data=(X_all[val_index,], Y_all[val_index,]))

            acc_x[-1][0].extend(history.history.get("loss"))
            acc_x[-1][1].extend(history.history.get("val_loss"))
            acc_x[-1][2].extend(history.history.get("acc"))
            acc_x[-1][3].extend(history.history.get("val_acc"))

            for i in range(nb_classes):
                class_acc_x[rerun,i] = model.evaluate(X_all[val_index,][y_all[val_index]==i],
                                                      Y_all[val_index,][y_all[val_index]==i],
                                                      verbose=0)[1]

        np.save("./acc_x_" + str(nb_samples) + '.npy', np.asarray(acc_x))
        np.save("./class_acc_x_" + str(nb_samples) + '.npy', np.asarray(class_acc_x))

    val_index_length = X_all.shape[0]/nb_clusters   # 20% of the dataset
    #train for source-inclusive
    for rerun in range(nb_rerun):

        val_index = []
        train_index = []

        # divide dataset accourding to exclusive source-aware partitioning

        for i in range(nb_classes):
            for j in range(nb_clusters):

                select = np.where(np.logical_and(est%nb_classes == i,est/nb_classes == j))[0]
                select = select[np.random.permutation(select.shape[0])]

                if nb_samples == 60000:
                    train_select = select[(val_index_length/(nb_classes*nb_clusters))::]
                    val_select = select[0:(val_index_length/(nb_classes*nb_clusters))]
                else:
                    train_select = select[0:(nb_samples/(nb_classes*nb_clusters))]
                    val_select = select[(nb_samples/(nb_classes*nb_clusters))::]

                train_index.extend(train_select)
                val_index.extend(val_select)

        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        val_index = val_index[0:val_index_length]

        if model_type == 'CNN':
            model = define_cnn(input_shape, nb_classes, cnn_type)
        elif model_type == 'MLP':
            model = define_mlp(input_shape, nb_classes)
        elif model_type == 'SVM':
            model = svm.SVC(C=1,gamma=0.01)

        if model_type == 'SVM':
            model.fit(X_all[train_index,], y_all[train_index])

            expected = y_all[val_index]
            predicted = model.predict(X_all[val_index,])

            acc_i.append(metrics.accuracy_score(expected, predicted))
            for i in range(nb_classes):
                class_acc_i[rerun,i] = metrics.accuracy_score(expected[y_all[val_index]==i],
                                                              predicted[y_all[val_index]==i])

        else:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

            acc_i.append([[],[],[],[]])
            history = model.fit(X_all[train_index,], Y_all[train_index,],
                                batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                                validation_data=(X_all[val_index,], Y_all[val_index,]))

            acc_i[-1][0].extend(history.history.get("loss"))
            acc_i[-1][1].extend(history.history.get("val_loss"))
            acc_i[-1][2].extend(history.history.get("acc"))
            acc_i[-1][3].extend(history.history.get("val_acc"))

            for i in range(nb_classes):
                class_acc_i[rerun,i] = model.evaluate(X_all[val_index,][y_all[val_index]==i],
                                                      Y_all[val_index,][y_all[val_index]==i],
                                                      verbose=0)[1]

        np.save("./acc_i_" + str(nb_samples) + '.npy', np.asarray(acc_i))
        np.save("./class_acc_i_" + str(nb_samples) + '.npy', np.asarray(class_acc_i))

    end = time.time()

    print(end-start)
