from __future__ import print_function
import numpy as np
import scipy.io as sio
import pandas as pd


def get_pseudo_labels_comlete(df, nb_metagenomes=75, nb_pseudos=1000):

    nb_sequences = df.shape[0]/nb_metagenomes
    X = np.zeros((nb_pseudos*nb_metagenomes,nb_sequences, df.shape[1]))

    count = 0
    order = np.arange(nb_sequences,dtype='int')
    for j in range(nb_metagenomes):
        a = df[df.sample_id==j].values
        X[count,:,:] = a[order,:]
        X[count,:,1] = 0
        count += 1
        print("Progress {:2.2%}".format((count)/float(nb_metagenomes*nb_pseudos)), end="\r")

    for i in range(1, nb_pseudos):
        order = np.random.permutation(nb_sequences)
        for j in range(nb_metagenomes):
            a = df[df.sample_id==j].values
            X[count,:,:] = a[order,:]
            X[count,:,1] = i
            count += 1
            print("Progress {:2.2%}".format((count)/float(nb_metagenomes*nb_pseudos)), end="\r")

    return X


def get_pseudo_labels_mini(df, nb_metagenomes=75, miniseqs_size=100, nb_pseudos=1000, replacement=True):

    nb_sequences = df.shape[0]/nb_metagenomes

    if replacement:
        order = np.random.randint(0,nb_sequences,miniseqs_size*nb_pseudos)
    else:
        order = np.random.permutation(nb_sequences)

    X = np.zeros((nb_pseudos*nb_metagenomes,miniseqs_size, df.shape[1]))
    count = 0
    for i in np.arange(nb_metagenomes):
        a = df[df.sample_id==i].values
        for j in np.arange(0,miniseqs_size*nb_pseudos,miniseqs_size):
            X[count,:,:] = a[order[j:j+miniseqs_size],:]
            X[count,:,1] = j/miniseqs_size
            count += 1
        print("Progress {:2.2%}".format((count)/float(nb_metagenomes*nb_pseudos)), end="\r")

    np.random.shuffle(X)

    X_train = X[:,:,[3,4]].reshape(X.shape[0], X.shape[1]*2)
    X_train = X_train.astype('float32')

    #normalize the input
    X_train = (X_train)/21.
    X_train -= X_train.mean(axis=0)
    X_train /= X_train.std(axis=0)

    y_train = X[:,0,1].astype('int')
    sample_ids = X[:,0,0].astype('int')

    return X_train, y_train, sample_ids 


def get_parent_labels_wrt_gene_call(df, nb_metagenomes=75):

    #creates parent labels depending on gene_ids.
    #applies zero padding from shorter genes
    #number of parents becomes 798

    gene_calls = pd.factorize(df.corresponding_gene_call)[1]
    nb_sequences = df.shape[0]/(nb_metagenomes*len(gene_calls))
    nb_parents = len(gene_calls)
    X = np.zeros((nb_parents*nb_metagenomes,nb_sequences, df.shape[1]))

    count = 0
    k = 0
    for i in gene_calls:
        foo = df[df.corresponding_gene_call==i]
        for j in range(nb_metagenomes):
            a = foo[foo.sample_id==j].values
            X[count,:,:] = a
            X[count,:,1] = k
            count += 1
            print("Progress {:2.2%}".format((count)/float(nb_metagenomes*nb_parents)), end="\r")
        k += 1

    return X
