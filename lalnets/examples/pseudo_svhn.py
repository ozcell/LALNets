from __future__ import print_function
                                
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K

from lalnets.commons.datasets import load_svhn
from lalnets.commons.visualizations import plot_class_means
from lalnets.commons.utils import *
from lalnets.commons.tsne import tsne
from lalnets.acol.models import define_cnn
from lalnets.acol.trainings import train_with_parents, train_with_pseudos, train_semisupervised
from sklearn.decomposition import PCA

import scipy as sc
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

np.random.seed(1337)  # for reproducibility
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sc.stats.sem(a)
    h = se * sc.stats.t._ppf((1+confidence)/2., n-1)
    return m, h
    
#load SVHN
(X_train, y_train), (X_test, y_test), input_shape, (_, _), = load_svhn('th', path='/home/ok18/Downloads/', extra=False)

X_train -= X_train.mean(axis=0)
X_test -= X_test.mean(axis=0)
#X_extra -= X_extra.mean(axis=0)
X_train = sample_wise_center_norm(X_train)
X_test = sample_wise_center_norm(X_test)
#X_extra = feature_wise_normalization(X_extra)

def get_pseudos(X, gen_type=0):
    X_out = np.copy(X)
    X_out = np.moveaxis(X_out, [2,3],[0,1])
    if gen_type == 0:
         X_out = X_out
    elif gen_type == 1:
         X_out = np.rot90(X_out,1)
    elif gen_type == 2:
         X_out = np.rot90(X_out,2)
    elif gen_type == 3:
         X_out = np.rot90(X_out,3)
    elif gen_type == 4:
         X_out = np.fliplr(X_out)
    elif gen_type == 5:
         X_out = np.fliplr(np.rot90(X_out,1))
    elif gen_type == 6:
         X_out = np.fliplr(np.rot90(X_out,2))
    elif gen_type == 7:
         X_out = np.fliplr(np.rot90(X_out,3))
    
    X_out = np.moveaxis(X_out, [2,3],[0,1])
                  
    return X_out

#training parameters
nb_reruns = 1
nb_epoch = 400
nb_dpoints = 80
batch_size = 400
nb_exp = 1

nb_pseudos = 8

#network parameters
conv_params = (32, 3, 2)  #(nb_filters, nb_conv, nb_pool)
cnn_type = 2
hidden_drop = True

#ACOL parameters
nb_clusters_per_pseudo = 20
p = 0.
c1 = 0.1
c2 = 1
c3 = 0.00000
c4 = 0.000001
balance_type=7
pooling = 'average'
trainable = False
acol_params = (nb_clusters_per_pseudo, p, c1, c2, c3, c4, balance_type, pooling, trainable)

#define optimizer
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)

#define model
model_params = (input_shape, nb_pseudos, cnn_type, conv_params, hidden_drop, acol_params)

metrics_all = []

est_train = []
output_pca_train = []

est_test = []
output_pca_test = []
output_test = []

output2 = np.zeros((len(X_train), 2048))
output = np.zeros((len(X_test), 2048))

save_to_path = None#'../../temp/exp_svhn_ps8k20_cnn2_32_7_e1_1_e000001_4000_adam/'

#if not os.path.exists(save_to_path):
#    os.makedirs(save_to_path)

for exp in range(nb_exp):

    adam = Adam()
    
    metrics, acti, model = train_with_pseudos(nb_pseudos = nb_pseudos, 
                                                nb_clusters_per_pseudo = nb_clusters_per_pseudo,
                                                define_model = define_cnn, 
                                                model_params = model_params, 
                                                optimizer = adam,
                                                X_train = X_train,  
                                                y_train = y_train,
                                                X_test = X_test, 
                                                y_test = y_test,
                                                get_pseudos = get_pseudos,
                                                nb_reruns = nb_reruns, 
                                                nb_epoch = nb_epoch,
                                                nb_dpoints = nb_dpoints, 
                                                batch_size = batch_size,
                                                test_on_test_set = False, 
                                                update_c3 = None,
                                                set_original_only = None,
                                                return_model = True,
                                                verbose = 1,
                                                save_after_each_rerun=None,
                                                model_in=None)

    metrics_all.append(metrics)
    
    model_truncated = model.get_model_truncated(define_cnn, model_params, nb_pseudos)
    
    est_test.append(model_truncated.predict(X_test))
    
    mypca = PCA(n_components=32)
    est_train.append(mypca.fit_transform(model_truncated.predict(X_train)))    
    
    get_metrics = K.function([model.layers[0].input, K.learning_phase()],
                            [model.layers[9].output])
  
    for batch_ind in range(0, len(X_test), 100):
        X_batch = X_test[batch_ind:batch_ind+100,]
        output[batch_ind:batch_ind+100,:] = get_metrics([X_batch,0])[0]

    output_test.append(output)
    
    mypca = PCA(n_components=32)
    pca_1 = mypca.fit_transform(output)
    
    output_pca_test.append(pca_1)
    
    for batch_ind in range(0, len(X_train), 100):
        X_batch = X_train[batch_ind:batch_ind+100,]
        output2[batch_ind:batch_ind+100,:] = get_metrics([X_batch,0])[0]
        
    mypca = PCA(n_components=32)
    pca_2 = mypca.fit_transform(output2)
    
    output_pca_train.append(pca_2)
    
    np.save(save_to_path + 'est_test.npy', est_test)
    np.save(save_to_path + 'output_pca_test.npy', output_pca_test)
    
    np.save(save_to_path + 'metrics_all.npy', metrics_all)
    
    
    np.save(save_to_path + 'est_train.npy', est_train)
    np.save(save_to_path + 'output_pca_train.npy', output_pca_train)
    
nb_clusters=10

acc_output = []
nmi_output = []
for i in range(len(output_test)):
    estimator = KMeans(init='k-means++', n_clusters=nb_clusters, n_init=nb_clusters)
    estimator.fit(output_test[i])
    acc_output.append(calculate_cl_acc(y_test, estimator.labels_,10,label_correction=True)[0])
    nmi_output.append(nmi(y_test, estimator.labels_))
    
np.save(save_to_path + 'cl_vacc.npy', acc_output)
np.save(save_to_path + 'cl_nmi.npy', nmi_output)

np.save(save_to_path + 'output_test.npy', output_test)
