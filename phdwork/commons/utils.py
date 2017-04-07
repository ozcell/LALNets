'''
Essential functions commonly used in scripts.

'''

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def plot_class_means(nb_classes, nb_clusters, est, test, cmap=None, boundary=None, sort=False, transpose=False, ax=None):
    if ax is None:
        ax = plt.gca()

    n,h,w,d = test.shape
    if boundary is None:
        boundary = range(n)

    if cmap is None:
        cmap = cm.jet

    if transpose:
        img = np.zeros((h*nb_classes,w*nb_clusters))
    else:
        img = np.zeros((h*nb_clusters,w*nb_classes))
    imgSorted = np.zeros_like(img)
    ind = np.zeros((nb_clusters, nb_classes))
    for i in range(nb_clusters):
        for j in range(nb_classes):
            a = test[boundary,:][est[boundary]==j+nb_classes*i]
            b = a.shape[0]
            if b == 0:
                a = np.zeros((h,w))
            else:
                a = a.mean(axis=0).reshape(h,w)
            ind[i,j] = b
            if transpose:
                img[j*h:(j+1)*h,i*w:(i+1)*w] = a
            else:
                img[i*h:(i+1)*h,j*w:(j+1)*w] = a

    for i in range(nb_clusters):
        for j in range(nb_classes):
            if sort:
                k = ind.argsort(axis=0)[::-1][i,j]
            else:
                k = i
            if transpose:
                imgSorted[j*h:(j+1)*h,i*w:(i+1)*w] = img[j*h:(j+1)*h,k*w:(k+1)*w]
            else:
                imgSorted[i*h:(i+1)*h,j*w:(j+1)*w] = img[k*h:(k+1)*h,j*w:(j+1)*w]

    fig = ax.imshow(imgSorted,cmap=cmap, interpolation='bicubic')

    ax.axis('off')
    ax.grid('off')

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
