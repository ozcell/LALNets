import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_class_means(nb_parents, nb_clusters_per_parent, est, test,
                     cmap=None, boundary=None, sort=False, transpose=False, ax=None, dim_ordering='th'):
    if ax is None:
        ax = plt.gca()
    if dim_ordering == 'tf':
        n,h,w,d = test.shape
    elif dim_ordering == 'th':
        n,d,h,w = test.shape

    if boundary is None:
        boundary = range(n)

    if cmap is None:
        cmap = cm.jet

    if transpose:
        img = np.zeros((h*nb_parents, w*nb_clusters_per_parent))
    else:
        img = np.zeros((h*nb_clusters_per_parent, w*nb_parents))
    imgSorted = np.zeros_like(img)
    ind = np.zeros((nb_clusters_per_parent, nb_parents))
    for i in range(nb_clusters_per_parent):
        for j in range(nb_parents):
            a = test[boundary, :][est[boundary] == j+nb_parents*i]
            b = a.shape[0]
            if b == 0:
                a = np.zeros((h, w))
            else:
                a = a.mean(axis=0).reshape(h, w)
            ind[i,j] = b
            if transpose:
                img[j*h:(j+1)*h, i*w:(i+1)*w] = a
            else:
                img[i*h:(i+1)*h, j*w:(j+1)*w] = a

    for i in range(nb_clusters_per_parent):
        for j in range(nb_parents):
            if sort:
                k = ind.argsort(axis=0)[::-1][i, j]
            else:
                k = i
            if transpose:
                imgSorted[j*h:(j+1)*h, i*w:(i+1)*w] = img[j*h:(j+1)*h, k*w:(k+1)*w]
            else:
                imgSorted[i*h:(i+1)*h, j*w:(j+1)*w] = img[k*h:(k+1)*h, j*w:(j+1)*w]

    fig = ax.imshow(imgSorted, cmap=cmap, interpolation='bicubic')

    ax.axis('off')
    ax.grid('off')
