import numpy as np
import time
from keras import backend as K
from keras.engine import Model
from keras.utils import np_utils, generic_utils
from phdwork.commons.utils import calculate_cl_acc, cumulate_metrics

'''

Functions to train and evaluate ACOL experiments.

'''

def train_with_parents(nb_parents, nb_clusters_per_parent,
                       define_model, model_params, optimizer,
                       X_train, y_train, y_train_parent,
                       X_test, y_test, y_test_parent,
                       nb_reruns, nb_epoch, nb_dpoints, batch_size,
                       test_on_test_set=True, update_c3=None,
                       return_model=False, verbose=1):

    #find the values of the dependent variables used inside the script
    nb_all_clusters = nb_parents*nb_clusters_per_parent

    # y to Y conversion for original dataset
    nb_classes = y_train.max() - y_train.min() + 1
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_train_parent = np_utils.to_categorical(y_train_parent, nb_parents)
    Y_test_parent = np_utils.to_categorical(y_test_parent, nb_parents)

    nb_epoch_per_dpoint = nb_epoch/nb_dpoints

    metrics = initialize_metrics()

    acti_train = np.zeros((len(y_train), nb_all_clusters, nb_reruns))
    acti_test = np.zeros((len(y_test), nb_all_clusters, nb_reruns))

    if test_on_test_set:
        test_data=(X_test, Y_test_parent)
    else:
        test_data=None

    for rerun in range(nb_reruns):

        rerun_start = time.time()

        #extend each list for each rerun
        for item in metrics.itervalues():
            item.append([])

        #add truncation info
        _model_params = model_params + (False,)
        _model_truncated_params = model_params + (True,)

        #define models for each run
        model = define_model(*_model_params)
        model_truncated = define_model(*_model_truncated_params)

        #and compile
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        model_truncated.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        #define a Theano function to reach values of ACOL metrics
        get_metrics = model.define_get_metrics()

        #test initial network before starting the training
        history = model.evaluate(X_train, Y_train_parent, batch_size, verbose=0)
        if test_data is not None:
            history.extend(model.evaluate(test_data[0], test_data[1], batch_size, verbose=0))
        else:
            history.extend([0, 0])

        #get ACOL metrics, affinity, balance, coactivity and regularization cost
        acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

        #calculate clustering accuracy
        cl_acc = model_truncated.evaluate_clustering(X_train, y_train, nb_all_clusters, batch_size, verbose=verbose)
        cl_vacc = model_truncated.evaluate_clustering(X_test, y_test, nb_all_clusters, batch_size, verbose=verbose)

        #update experiment metrics
        update_metrics(metrics, history, [cl_acc, cl_vacc], acol_metrics)

        #print stats
        print_stats(verbose, 1, test_data=test_data, X_train=X_train,
                    nb_pseudos=nb_pseudos, acol_metrics=acol_metrics,
                    cl_acc=[cl_acc, cl_vacc])

        for dpoint in range(nb_dpoints):

            history = model.fit(X_train, Y_train_parent,
                                batch_size=batch_size, nb_epoch=nb_epoch_per_dpoint,
                                verbose=2, test_data=test_data)

            history = history.history.values()
            history = [history[1][-1],history[0][-1],history[3][-1],history[2][-1]]

            #transfer weights to truncated mirror of the model
            model_truncated.set_weights(model.get_weights())

            #get ACOL metrics, affinity, balance, coactivity and regularization cost
            acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

            #calculate clustering accuracy
            cl_acc = model_truncated.evaluate_clustering(X_train, y_train, nb_all_clusters, batch_size, verbose=verbose)
            cl_vacc = model_truncated.evaluate_clustering(X_test, y_test, nb_all_clusters, batch_size, verbose=verbose)

            #ACOL c3 update
            if update_c3 is not None:
                new_c3 = update_c3(acol_metrics, (dpoint+1)*nb_epoch_per_dpoint, verbose=verbose)
                model.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)
                model_truncated.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)

            #update experiment metrics
            update_metrics(metrics, history, [cl_acc, cl_vacc], acol_metrics)

            #print stats
            print_stats(verbose, 2, test_data=test_data, rerun=rerun,
                        dpoint=dpoint, nb_epoch_per_dpoint=nb_epoch_per_dpoint,
                        acol_metrics=acol_metrics, cl_acc=[cl_acc, cl_vacc])

        acti_train[:,:,rerun] = model_truncated.predict(X_train, batch_size=batch_size)
        acti_test[:,:,rerun] = model_truncated.predict(X_test, batch_size=batch_size)

        rerun_end = time.time()

        #print stats
        print_stats(verbose, 3, rerun_start=rerun_start, rerun_end=rerun_end,
                    nb_reruns=nb_reruns, rerun=rerun)

    return metrics, (acti_train, acti_test), model if return_model else None


def train_with_pseudos(nb_pseudos, nb_clusters_per_pseudo,
                       define_model, model_params, optimizer,
                       X_train, y_train,
                       X_test, y_test,
                       get_pseudos,
                       nb_reruns, nb_epoch, nb_dpoints, batch_size,
                       test_on_test_set=True, update_c3=None,
                       set_original_only=None, return_model=False,
                       verbose=1):

    #find the values of the dependent variables used inside the script
    nb_all_clusters = nb_pseudos*nb_clusters_per_pseudo

    # y to Y conversion for original dataset
    nb_classes = y_train.max() - y_train.min() + 1
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    nb_epoch_per_dpoint = nb_epoch/nb_dpoints

    metrics = initialize_metrics()

    acti_train = np.zeros((len(y_train), nb_all_clusters, nb_reruns))
    acti_test = np.zeros((len(y_test), nb_all_clusters, nb_reruns))

    if test_on_test_set:
        test_data=(X_test, )
    else:
        test_data=None

    for rerun in range(nb_reruns):

        rerun_start = time.time()

        #extend each list for each rerun
        for item in metrics.itervalues():
            item.append([])

        #add truncation info
        _model_params = model_params + (False,)
        _model_truncated_params = model_params + (True,)

        #define models for each run
        model = define_model(*_model_params)
        model_truncated = define_model(*_model_truncated_params)

        #and compile
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        model_truncated.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        #train only using the original dataset i.e. X^*(0)
        original_only = False

        #define a Theano function to reach values of ACOL metrics
        get_metrics = model.define_get_metrics()

        #test initial network before starting the training
        history = model.fit_pseudo(X_train, nb_pseudos,
                                batch_size=batch_size, nb_epoch=nb_epoch_per_dpoint, train=False,
                                get_pseudos=get_pseudos, test_data=test_data,
                                train_on_original_only=original_only, verbose=0)

        #get ACOL metrics, affinity, balance, coactivity and regularization cost
        acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

        #calculate clustering accuracy
        cl_acc = model_truncated.evaluate_clustering(X_train, y_train, nb_all_clusters, batch_size, verbose=verbose)
        cl_vacc = model_truncated.evaluate_clustering(X_test, y_test, nb_all_clusters, batch_size, verbose=verbose)

        #update experiment metrics
        update_metrics(metrics, history, [cl_acc, cl_vacc], acol_metrics)

        #print stats
        print_stats(verbose, 1, test_data=test_data, X_train=X_train,
                    nb_pseudos=nb_pseudos, acol_metrics=acol_metrics,
                    cl_acc=[cl_acc, cl_vacc])

        for dpoint in range(nb_dpoints):

            history = model.fit_pseudo(X_train, nb_pseudos,
                                batch_size=batch_size, nb_epoch=nb_epoch_per_dpoint, train=True,
                                get_pseudos=get_pseudos, test_data=test_data,
                                train_on_original_only=original_only, verbose=verbose)

            #transfer weights to truncated mirror of the model
            model_truncated.set_weights(model.get_weights())

            #get ACOL metrics, affinity, balance, coactivity and regularization cost
            acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

            #calculate clustering accuracy
            cl_acc = model_truncated.evaluate_clustering(X_train, y_train, nb_all_clusters, batch_size, verbose=verbose)
            cl_vacc = model_truncated.evaluate_clustering(X_test, y_test, nb_all_clusters, batch_size, verbose=verbose)

            #ACOL c3 update
            if update_c3 is not None:
                new_c3 = update_c3(acol_metrics, (dpoint+1)*nb_epoch_per_dpoint, verbose=verbose)
                model.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)
                model_truncated.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)

            if set_original_only is not None:
                original_only = set_original_only(acol_metrics, (dpoint+1)*nb_epoch_per_dpoint, verbose=verbose)

            #update experiment metrics
            update_metrics(metrics, history, [cl_acc, cl_vacc], acol_metrics)

            #print stats
            print_stats(verbose, 2, test_data=test_data, rerun=rerun,
                        dpoint=dpoint, nb_epoch_per_dpoint=nb_epoch_per_dpoint,
                        acol_metrics=acol_metrics, cl_acc=[cl_acc, cl_vacc])

        acti_train[:,:,rerun] = model_truncated.predict(X_train, batch_size=batch_size)
        acti_test[:,:,rerun] = model_truncated.predict(X_test, batch_size=batch_size)

        rerun_end = time.time()

        #print stats
        print_stats(verbose, 3, rerun_start=rerun_start, rerun_end=rerun_end,
                    nb_reruns=nb_reruns, rerun=rerun)

    return metrics, (acti_train, acti_test), model if return_model else None


def fit_pseudo(self, X_train, nb_pseudos, batch_size, nb_epoch,
               get_pseudos, train=True, test_data=None,
               train_on_original_only=False, test_on_original_only=True, verbose=1):

    if verbose:
        progbar = generic_utils.Progbar(nb_epoch)

    for epoch in range(nb_epoch):

        if train:
            #train model
            for X_batch, Y_batch in pseudo_batch_generator(X_train, batch_size,
                nb_pseudos, get_pseudos, train_on_original_only):
                self.train_on_batch(X_batch, Y_batch)

        #test on original training set
        count = 0
        history_train = np.zeros(2)
        for X_batch, Y_batch in pseudo_batch_generator(X_train, batch_size,
            nb_pseudos, get_pseudos, test_on_original_only):
            history_train += self.test_on_batch(X_batch, Y_batch)
            count += 1
        history_train /= count
        history_train = list(history_train)
        values=[('loss', history_train[0]), ('acc', history_train[1])]

        #test on original test set if test_on_test_set is True
        history_test = np.zeros(2)
        if test_data is not None:
            count = 0
            for X_batch, Y_batch in pseudo_batch_generator(test_data[0],
                batch_size, nb_pseudos, get_pseudos, test_on_original_only):
                history_test += self.test_on_batch(X_batch, Y_batch)
                count += 1
            history_test /= count
            history_test = list(history_test)
            values.extend([('val_loss', history_test[0]), ('val_acc', history_test[1])])
            history_train.extend(history_test)
        else:
            #values.extend([('val_loss', 'N/A'), ('val_acc', 'N/A')])
            history_train.extend(history_test)

        if verbose:
            progbar.add(1, values=values)

    return history_train


Model.fit_pseudo = fit_pseudo


def pseudo_batch_generator(X, batch_size, nb_pseudos, get_pseudos, original_only):

    #initialize shuffled ind
    if original_only:
        ind = np.random.permutation(len(X))
    else:
        ind = np.random.permutation(nb_pseudos*len(X))

    for ind_batch_start in range(0, len(ind), batch_size):

        #if the last batch then the size is the number of remaining samples
        if ind_batch_start+batch_size > len(ind):
            ind_batch = ind[ind_batch_start::]
        else:
            ind_batch = ind[ind_batch_start:ind_batch_start+batch_size]

        #ind_batch%len(X) --> which sample
        #ind_batch/len(X) --> which pseudo label

        #take a batch of X depending of the remainder
        X_batch = X[ind_batch%len(X),]

        #create pseudo labels
        if original_only:
            y_batch = np.zeros_like(ind_batch) #create all zeros output labels
        else:
            y_batch = ind_batch/len(X) #create suffled eqaully distributed
                                             #labels with values 0 to nb_pseudos

        #in case if nb_pseudos=1 to support null_node
        if nb_pseudos > 1:
            Y_batch = np_utils.to_categorical(y_batch, nb_pseudos)
        else:
            Y_batch = np_utils.to_categorical(y_batch, nb_pseudos+1)

        #transform X according to pseudo labels
        for pseudo in range(nb_pseudos):
            X_batch[y_batch==pseudo,] = get_pseudos(X_batch[y_batch==pseudo,], pseudo)

        yield X_batch, Y_batch


def evaluate_clustering(self, X, y, nb_all_clusters, batch_size=128,
                    cluster_offset=0, label_correction=False, verbose=0):

    #calculate clustering accuracy
    est = self.predict_classes(X, batch_size=batch_size, verbose=0)
    cl_acc = calculate_cl_acc(y, est, nb_all_clusters, cluster_offset, label_correction)

    #Check if clustering accuracy is calculated over entire dataset
    if verbose == 1:
        if cl_acc[1] != len(X):
            print("!" * 40)
            print('Warning! Check cluster accuracy calcualtions. Consider label_correction.')
            print("!" * 40)

    return cl_acc[0]


Model.evaluate_clustering = evaluate_clustering


def define_get_metrics(self):

    #define a Theano function to reach values of ACOL metrics
    get_metrics = K.function([self.layers[0].input, K.learning_phase()],
                             [self.get_layer("L-1").activity_regularizer.affinity,
                              self.get_layer("L-1").activity_regularizer.balance,
                              self.get_layer("L-1").activity_regularizer.coactivity,
                              self.get_layer("L-1").activity_regularizer.reg])

    return get_metrics


Model.define_get_metrics = define_get_metrics


def update_metrics(metrics, history, cl_acc, acol_metrics):

    metrics.get('loss')[-1].append(history[0])
    metrics.get('acc')[-1].append(history[1])
    metrics.get('vloss')[-1].append(history[2])
    metrics.get('vacc')[-1].append(history[3])

    metrics.get('cl_acc')[-1].append(cl_acc[0])
    metrics.get('cl_vacc')[-1].append(cl_acc[1])

    metrics.get('affinity')[-1].append(acol_metrics[0])
    metrics.get('balance')[-1].append(acol_metrics[1])
    metrics.get('coactivity')[-1].append(acol_metrics[2])
    metrics.get('reg')[-1].append(acol_metrics[3])


def initialize_metrics():

    #initialize evaluation metrics
    loss, vloss = [], []
    acc, vacc = [], []
    cl_acc, cl_vacc = [], []
    affinity, balance, coactivity, reg  = [], [], [], []
    metrics = {'loss': loss,
               'vloss': vloss,
               'acc': acc,
               'vacc': vacc,
               'cl_acc': cl_acc,
               'cl_vacc': cl_vacc,
               'affinity': affinity,
               'balance': balance,
               'coactivity': coactivity,
               'reg': reg}

    return metrics

def print_stats(verbose, stat_type, **kwargs) :

    if verbose:

        if stat_type == 1:

            test_data = kwargs.get('test_data')
            X_train = kwargs.get('X_train')
            nb_pseudos = kwargs.get('nb_pseudos')
            if X_train is not None and nb_pseudos is not None:
                if test_data is not None:
                    print('Train on %d samples, test on %d samples' %
                          (len(X_train)*nb_pseudos, len(test_data[0])))
                else:
                    print('Train on %d samples' % (len(X_train)*nb_pseudos))

            acol_metrics = kwargs.get('acol_metrics')
            cl_acc = kwargs.get('cl_acc')
            print("*" * 40)
            if acol_metrics is not None:
                print('Stats before training:')
                print('ACOL metrics: Affinity: %.3f, Balance: %.3f, Coactivity: %.3f' %
                     (acol_metrics[0], acol_metrics[1], acol_metrics[2]))
            if cl_acc is not None:
                print('Clustering accuracy: On training set: %.3f, On test set: %.3f' %
                     (cl_acc[0], cl_acc[1]))
            print("*" * 40)

        elif stat_type == 2:
            acol_metrics = kwargs.get('acol_metrics')
            cl_acc = kwargs.get('cl_acc')
            rerun = kwargs.get('rerun')
            dpoint = kwargs.get('dpoint')
            nb_epoch_per_dpoint = kwargs.get('nb_epoch_per_dpoint')
            print("*" * 40)
            if rerun is not None and dpoint is not None and nb_epoch_per_dpoint is not None:
                print('Stats at epoch ' + str((dpoint+1)*nb_epoch_per_dpoint) + ' of rerun ' + str(rerun+1))
            if acol_metrics is not None:
                print('ACOL metrics: Affinity: %.3f, Balance: %.3f, Coactivity: %.3f' %
                     (acol_metrics[0], acol_metrics[1], acol_metrics[2]))
            if cl_acc is not None:
                print('Clustering accuracy: On training set: %.3f, On test set: %.3f' %
                     (cl_acc[0], cl_acc[1]))
            print("*" * 40)

        elif stat_type == 3:
            rerun_start = kwargs.get('rerun_start')
            rerun_end = kwargs.get('rerun_end')
            nb_reruns = kwargs.get('nb_reruns')
            rerun = kwargs.get('rerun')
            print("*" * 40)
            if rerun_start is not None and rerun_end is not None and nb_reruns is not None and rerun is not None:
                print('Estimated remaining run time: ' + str(int((rerun_end-rerun_start)*(nb_reruns-(rerun+1)))) + ' sec')
            print("*" * 40)
