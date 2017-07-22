'''
Essential functions commonly used in scripts.

'''

import numpy as np
import pandas as pd


def combine_classes(y, nb_parents):

    nb_classes = y.max() - y.min() + 1
    ratio = nb_classes/nb_parents

    y_parent = np.zeros_like(y)

    for parent in range(nb_parents):
        y_parent[y/ratio == parent] = parent

    return y_parent


def calculate_cl_acc(ground_truth, est, nb_all_clusters, cluster_offset=0, label_correction=False):

    majority = np.zeros(nb_all_clusters)
    population = np.zeros(nb_all_clusters)

    if label_correction:
        est = correct_labels(ground_truth, est)

    for cluster in range(cluster_offset, nb_all_clusters + cluster_offset):
        if np.bincount(ground_truth[est==cluster]).size != 0:
            majority[cluster-cluster_offset] = np.bincount(ground_truth[est==cluster]).max()
            population[cluster-cluster_offset] = np.bincount(ground_truth[est==cluster]).sum()

    cl_acc = majority[majority>0].sum()/population[population>0].sum()

    return cl_acc, population.sum()


def correct_labels(ground_truth, est):

    corrested_est = np.zeros_like(est, dtype='int')

    for cluster in range(est.max()+1):
        if np.bincount(ground_truth[est==cluster]).size != 0:
            true_label = np.bincount(ground_truth[est==cluster]).argmax()
            corrested_est[est==cluster] = true_label

    return corrested_est


def cumulate_metrics(X, metric_func, batch_size=128):

    count = 0

    metrics = np.zeros(metric_func.function.n_returned_outputs)

    for batch_ind in range(0, len(X), batch_size):
        X_batch = X[batch_ind:batch_ind+batch_size,]
        metrics += metric_func([X_batch,0])
        count += 1

    metrics /= count

    return metrics

def get_acception_activations(acti, nb_parents, acception_type='cluster_wise'):

    #get activation shape
    nb_samples, nb_all_clusters, nb_reruns = acti.shape
    nb_clusters_per_parent = nb_all_clusters/nb_parents

    #initialize return variables
    acti_clusters = []
    for i in range(nb_clusters_per_parent):
        acti_clusters.append([])

    #reshape activation
    acti = acti.swapaxes(1,2).reshape(nb_samples*nb_reruns, nb_all_clusters)

    if acception_type == 'cluster_wise':
        for accepting_cluster in range(nb_all_clusters):

            #indices of samples accepted by accepting cluster
            accepted_ind = (acti.argmax(axis=1)==accepting_cluster)

            #parent of this cluster
            parent = accepting_cluster%nb_parents

            #list of all clusters of parent
            clusters = range(parent, nb_all_clusters, nb_parents)

            #list of rejecting clusters
            rejecting_clusters = range(parent, nb_all_clusters, nb_parents)
            rejecting_clusters.remove(accepting_cluster)

            acti_clusters[0].extend(acti[accepted_ind, accepting_cluster])
            for i in range(nb_clusters_per_parent-1):
                acti_clusters[i+1].extend(acti[accepted_ind, rejecting_clusters[i]])

    elif acception_type == 'parent_wise':
        for parent in range(nb_parents):

            #indices of samples accepted by accepting parent
            accepted_ind = (acti.argmax(axis=1)%nb_parents==parent)

            #list of all clusters of parent
            clusters = range(parent, nb_all_clusters, nb_parents)

            for i in range(nb_clusters_per_parent):
                acti_clusters[i].extend(acti[accepted_ind, clusters[i]])

    return acti_clusters

def choose_samples(X, y, nb_classes, nb_samples_per_class, verbose=1):

    perm = np.random.permutation
    ind_chosen= []
    for label in range(0, nb_classes):
        #select all samples of a labels
        ind_samples = np.where(y == label)[0]
        #shuffle them
        ind_samples = ind_samples[perm(len(ind_samples))]
        #choose first nb_samples_per_class samples and add to list
        ind_samples = ind_samples[0:nb_samples_per_class]
        ind_chosen.extend(ind_samples)

    if verbose > 1:
        print(ind_chosen)

    np.random.shuffle(ind_chosen)

    return X[ind_chosen, ], y[ind_chosen]

def sample_wise_center_norm(X):

    if len(X.shape) == 2:
        X -= X.mean(axis=(1), keepdims=True)
        X /= X.std(axis=(1), keepdims=True) + 1e-8
    elif len(X.shape) == 3:
        X -= X.mean(axis=(1,2), keepdims=True)
        X /= X.std(axis=(1,2), keepdims=True) + 1e-8
    elif len(X.shape) == 4:
        X -= X.mean(axis=(1,2,3), keepdims=True)
        X /= X.std(axis=(1,2,3), keepdims=True) + 1e-8

    return X

def get_similarity_matrix(sample_ids, est, nb_parents, nb_clusters_per_parent, weighted=False):

    nb_samples = sample_ids.max() - sample_ids.min() + 1

    a = np.zeros((nb_parents, nb_clusters_per_parent))
    for i in np.arange(est.shape[0]):
        k = est[i]%nb_parents
        l = est[i]/nb_parents
        a[k,l] += 1

    sim = np.zeros((nb_samples, nb_samples))
    b = sample_ids
    c = est
    for i in pd.factorize(np.sort(est))[1]:
        d = np.sort(b[c==i]).astype(int)
        if len(d)>0:
            if weighted:
                sim[d.reshape(len(d),1),d.reshape(1,len(d))] += 1/float(len(d))
            else:
                sim[d.reshape(len(d),1),d.reshape(1,len(d))] += 1.

    sim = np.triu(sim,1)
    sim = sim - sim[sim.nonzero()].min()
    sim = np.triu(sim,1)
    sim = sim / sim.max()
    sim = np.triu(sim,1)

    sim = np.triu(sim,1).T+np.triu(sim,1)

    for i in range(nb_samples):
        sim[i,i] = 1

    return a, sim


def get_nb_nonempty_clusterings(a):

    clusters = np.zeros(a.shape[1]-1)
    a.sort(axis=1)
    sum_higher = 0
    for j in range(a.shape[1]-1):
        clusters[j] = (a[:,j]!=0).sum() - sum_higher
        sum_higher += (a[:,j]!=0).sum() - sum_higher

    return clusters
