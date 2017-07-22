from __future__ import print_function

from keras.optimizers import SGD

from lalnets.commons.datasets import load_svhn
from lalnets.commons.utils import sample_wise_center_norm
from lalnets.acol.models import define_cnn
from lalnets.acol.trainings import train_semisupervised

import scipy as sc
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

np.random.seed(1337)  # for reproducibility
%matplotlib inline

#load SVHN
(X_train, y_train), (X_test, y_test), input_shape, (X_extra, y_extra), = load_svhn('th', extra=False)
#X_train -= X_train.mean(axis=0) # feature_wise centering
#X_test -= X_test.mean(axis=0) # feature_wise centering
X_train = sample_wise_center_norm(X_train)
X_test = sample_wise_center_norm(X_test)

save_to_path = '../temp/exp_svhn_c1s1000_cnn4_64_32_96/'

if not os.path.exists(save_to_path):
    os.makedirs(save_to_path)

#training parameters
nb_reruns = 10
nb_epoch = 100
nb_dpoints = 20
batch_size = 32 #b_U

#training parameters
nb_epoch_pre = 2000
batch_size_pre = 96 #b_L

#pseudo classes options
nb_pseudos = 1
def get_pseudos(X, gen_type=0):
    return X

#network parameters
conv_params = (64, 3, 2)  #(nb_filters, nb_conv, nb_pool)
cnn_type = 4
hidden_drop = False

#ACOL parameters
nb_clusters_per_pseudo = 10
p = 0.
c1 = 3 #c_alpha
c2 = 1 #c_beta
c3 = 0
c4 = 0.000001 #c_F
pooling = 'average'
trainable = False
acol_params = (nb_clusters_per_pseudo, p, c1, c2, c3, c4, pooling, trainable)

#define optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)

#define model
model_params = (input_shape, nb_pseudos, cnn_type, conv_params, hidden_drop, acol_params)

#define pre_model
model_pre_params = (True, 0., 0., 0., 0.)

#number of labeled samples
nb_labeled = 1000 #m_L

#define optimizer for pretraining
sgd_pre = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)

metrics, acti, model = train_semisupervised(nb_pseudos = nb_pseudos,
                                            nb_clusters_per_pseudo = nb_clusters_per_pseudo,
                                            define_model = define_cnn,
                                            model_params = (model_params, model_pre_params),
                                            optimizer = (sgd, sgd_pre),
                                            X_train = (X_train, X_train),
                                            y_train = (y_train, y_train),
                                            nb_labeled = nb_labeled,
                                            X_test = X_test,
                                            y_test = y_test,
                                            get_pseudos = get_pseudos,
                                            nb_reruns = nb_reruns,
                                            nb_epoch = (nb_epoch, nb_epoch_pre),
                                            nb_dpoints = nb_dpoints,
                                            batch_size = (batch_size, batch_size_pre),
                                            validation_set_size = 1000,
                                            test_on_test_set = False,
                                            update_c3 = None,
                                            set_original_only = None,
                                            return_model = True,
                                            verbose = 1,
                                            save_after_each_rerun=save_to_path,
                                            model_in=None)
