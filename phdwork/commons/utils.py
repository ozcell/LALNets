'''
Essential functions commonly used in scripts.

'''

import numpy as np


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

def get_acception_rejection_activations(acti, nb_parents):

    """
    Seperates the cluster activations into two group.

    Returns:

    acti_accepting_clusters: activation of the clusters observed wrt accepted samples
    acti_rejecting_clusters: activations all other rejecting clusters

    """

    #get activation shape
    nb_samples, nb_all_clusters, nb_reruns = acti.shape

    #initialize return variables
    acti_accepting_clusters, acti_rejecting_clusters = [], []

    #reshape activation
    acti = acti.swapaxes(1,2).reshape(nb_samples*nb_reruns, nb_all_clusters)

    for accepting_cluster in range(nb_all_clusters):

        #indices of samples accepted by accepting cluster
        accepted_ind = (acti.argmax(axis=1)==accepting_cluster)

        #accepting cluster belongs to this parent
        parent = accepting_cluster%nb_parents

        #list of rejecting clusters
        rejecting_clusters = range(parent, nb_all_clusters, nb_parents).remove(accepting_cluster)

        #activation observed on accepting cluster and rejecting clusters
        acti_accepting_clusters.extend(acti[accepted_ind, accepting_cluster])
        acti_rejecting_clusters.extend(acti[accepted_ind, rejecting_clusters])

    return (acti_accepting_clusters, acti_rejecting_clusters)
