'''
Essential functions commonly used in scripts.

'''

import numpy as np


def combine_classes(y, nb_parent_classes):

    nb_classes = y.max() - y.min() + 1
    ratio = nb_classes/nb_parent_classes

    y_parent = np.zeros_like(y)

    for label in range(nb_parent_classes):
        y_parent[y/ratio==label] = label

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

    for label in range(est.max()+1):
        if np.bincount(ground_truth[est==label]).size != 0:
            true_label = np.bincount(ground_truth[est==label]).argmax()
            corrested_est[est==label] = true_label

    return corrested_est


def cumulate_acol_metrics(X, metric_func, batch_size=128):

    affinity_sum, balance_sum, coactivity_sum, reg_sum, count = 0, 0, 0, 0, 0

    for batch_ind in range(0, len(X), batch_size):
        X_batch = X[batch_ind:batch_ind+batch_size,]

        get_metrics_list = metric_func([X_batch,0])

        affinity_sum += get_metrics_list[0].item()
        balance_sum += get_metrics_list[1].item()
        coactivity_sum += get_metrics_list[2].item()
        reg_sum += get_metrics_list[3].item()

        count += 1

    affinity_sum /= count
    balance_sum /= count
    coactivity_sum /= count
    reg_sum /= count

    return affinity_sum, balance_sum, coactivity_sum, coactivity_sum, reg_sum
