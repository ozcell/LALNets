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
                       model_def_func, model_params, optimizer,
                       X_train, y_train, y_train_parent,
                       X_test, y_test, y_test_parent,
                       nb_reruns, nb_epoch, nb_dpoints, batch_size,
                       validate_on_test_set=True, c3_update_func=None,
                       return_model=False):

    #find the values of the dependent variables used inside the script
    nb_all_clusters = nb_parents*nb_clusters_per_parent
    nb_classes = y_train.max() - y_train.min() + 1
    nb_epoch_per_dpoint = nb_epoch/nb_dpoints

    # y to Y conversion for original dataset
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # y to Y conversion for parent dataset
    Y_train_parent = np_utils.to_categorical(y_train_parent, nb_parents)
    Y_test_parent = np_utils.to_categorical(y_test_parent, nb_parents)

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

    #initialize activation matrices
    acti_train = np.zeros((len(X_train), nb_all_clusters, nb_reruns))
    acti_test = np.zeros((len(X_test), nb_all_clusters, nb_reruns))

    if validate_on_test_set:
        validation_data=(X_test, Y_test_parent)
    else:
        validation_data=None

    for rerun in range(nb_reruns):

        start = time.time()

        #extend each list for each rerun
        for item in metrics.itervalues():
            item.append([])

        #add truncation info
        _model_params = model_params + (False,)
        _model_truncated_params = model_params + (True,)

        #define models for each run
        model = model_def_func(*_model_params)
        model_truncated = model_def_func(*_model_truncated_params)

        #and compile
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        model_truncated.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        #define a Theano function to reach values of ACOL metrics
        get_metrics = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.get_layer("L-1").activity_regularizer.affinity,
                                  model.get_layer("L-1").activity_regularizer.balance,
                                  model.get_layer("L-1").activity_regularizer.coactivity,
                                  model.get_layer("L-1").activity_regularizer.reg])

        for dpoint in range(nb_dpoints):

            history = model.fit(X_train, Y_train_parent,
                                batch_size=batch_size, nb_epoch=nb_epoch_per_dpoint,
                                verbose=2, validation_data=validation_data)

            history = history.history.values()
            history = [history[1][-1],history[0][-1],history[3][-1],history[2][-1]]

            #transfer weights to truncated mirror of the model
            model_truncated.set_weights(model.get_weights())

            #get ACOL metrics, affinity, balance, coactivity and regularization cost
            acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

            #calculate clustering accuracy
            est_train = model_truncated.predict_classes(X_train, batch_size=batch_size, verbose=0)
            est_test = model_truncated.predict_classes(X_test, batch_size=batch_size, verbose=0)
            _cl_acc = calculate_cl_acc(y_train, est_train, nb_all_clusters, 0, False)
            _cl_vacc = calculate_cl_acc(y_test, est_test, nb_all_clusters, 0, False)

            #Check if clustering accuracy is calculated over entire dataset
            if (_cl_acc[1] != len(X_train)) or (_cl_vacc[1] != len(X_test)):
                print("!" * 40)
                print('Warning! Check cluster accuracy calcualtions. Consider label_correction.')
                print("!" * 40)

            #ACOL c3 update
            if c3_update_func is not None:
                new_c3 = c3_update_func(acol_metrics, (dpoint+1)*nb_epoch_per_dpoint, verbose = 0)
                model.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)
                model_truncated.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)

            metrics.get('loss')[-1].append(history[0])
            metrics.get('acc')[-1].append(history[1])
            metrics.get('vloss')[-1].append(history[2])
            metrics.get('vacc')[-1].append(history[3])

            metrics.get('cl_acc')[-1].append(_cl_acc[0])
            metrics.get('cl_vacc')[-1].append(_cl_vacc[0])

            metrics.get('affinity')[-1].append(acol_metrics[0])
            metrics.get('balance')[-1].append(acol_metrics[1])
            metrics.get('coactivity')[-1].append(acol_metrics[2])
            metrics.get('reg')[-1].append(acol_metrics[3])

            print("*" * 40)
            print('End of epoch ' + str((dpoint+1)*nb_epoch_per_dpoint) + ' of rerun ' + str(rerun+1))
            print("*" * 40)

        acti_train[:,:,rerun] = model_truncated.predict(X_train, batch_size=batch_size)
        acti_test[:,:,rerun] = model_truncated.predict(X_test, batch_size=batch_size)

        end = time.time()

        print("*" * 40)
        print('Estimated remaining run time: ' + str(int((end-start)*(nb_reruns-(rerun+1)))) + ' sec')
        print("*" * 40)

    return metrics, (acti_train, acti_test), model if return_model else None


def train_with_pseudos(nb_pseudos, nb_clusters_per_pseudo,
                       model_def_func, model_params, optimizer,
                       X_train, y_train,
                       X_test, y_test,
                       get_pseudos_func,
                       nb_reruns, nb_epoch, nb_dpoints, batch_size,
                       validate_on_test_set=True, c3_update_func=None,
                       set_only_original_func=None, return_model=False,
                       verbose=1):


    metrics, (Y_train, Y_test), (acti_train, acti_test), nb_all_clusters,
    nb_classes, nb_epoch_per_dpoint = initialize_training_variables(nb_pseudos,
                                nb_clusters_per_pseudo, y_train, y_test,
                                nb_reruns, nb_epoch, nb_dpoints)
                                
    if validate_on_test_set:
        validation_data=(X_test, )
    else:
        validation_data=None


    for rerun in range(nb_reruns):

        start = time.time()

        #extend each list for each rerun
        for item in metrics.itervalues():
            item.append([])

        #add truncation info
        _model_params = model_params + (False,)
        _model_truncated_params = model_params + (True,)

        #define models for each run
        model = model_def_func(*_model_params)
        model_truncated = model_def_func(*_model_truncated_params)

        #and compile
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        model_truncated.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        #train only using the original dataset i.e. X^*(0)
        only_original = False

        #define a Theano function to reach values of ACOL metrics
        get_metrics = model.define_get_metrics()

        #validate initial network before starting the training
        history = model.fit_pseudo(X_train, nb_pseudos,
                                batch_size=batch_size, nb_epoch=nb_epoch_per_dpoint, train=False,
                                get_pseudos_func=get_pseudos_func, validation_data=validation_data,
                                train_on_only_original=only_original)

        #get ACOL metrics, affinity, balance, coactivity and regularization cost
        acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

        #calculate clustering accuracy
        cl_acc = model_truncated.evaluate_clustering(X_train, y_train, nb_all_clusters, batch_size)
        cl_vacc = model_truncated.evaluate_clustering(X_test, y_test, nb_all_clusters, batch_size)

        #update experiment metrics
        update_metrics(metrics, history, [cl_acc, cl_vacc], acol_metrics)

        if verbose == 1:
            progbar = generic_utils.Progbar(nb_dpoints*nb_epoch_per_dpoint)
            values=[('a', acol_metrics[0]), ('b', acol_metrics[1]),
                    ('c', acol_metrics[2])]

            progbar.add(0, values=values)

        for dpoint in range(nb_dpoints):

            history = model.fit_pseudo(X_train, nb_pseudos,
                                batch_size=batch_size, nb_epoch=nb_epoch_per_dpoint, train=True,
                                get_pseudos_func=get_pseudos_func, validation_data=validation_data,
                                train_on_only_original=only_original)

            #transfer weights to truncated mirror of the model
            model_truncated.set_weights(model.get_weights())

            #get ACOL metrics, affinity, balance, coactivity and regularization cost
            acol_metrics = cumulate_metrics(X_train, get_metrics, batch_size)

            #calculate clustering accuracy
            cl_acc = model_truncated.evaluate_clustering(X_train, y_train, nb_all_clusters, batch_size)
            cl_vacc = model_truncated.evaluate_clustering(X_test, y_test, nb_all_clusters, batch_size)

            #ACOL c3 update
            if c3_update_func is not None:
                new_c3 = c3_update_func(acol_metrics, (dpoint+1)*nb_epoch_per_dpoint, verbose = 0)
                model.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)
                model_truncated.get_layer("L-1").activity_regularizer.c3.set_value(new_c3)

            if set_only_original_func is not None:
                only_original = set_only_original_func(acol_metrics, (dpoint+1)*nb_epoch_per_dpoint, verbose = 0)

            #update experiment metrics
            update_metrics(metrics, history, [cl_acc, cl_vacc], acol_metrics)

            if verbose == 1:
                values=[('ca', cl_acc[0]), ('cva', cl_vacc[0]),
                        ('a', acol_metrics[0]), ('b', acol_metrics[1]),
                        ('c', acol_metrics[2])]

                progbar.add(nb_epoch_per_dpoint, values=values)
            elif verbose == 2:
                print("*" * 40)
                print('End of epoch ' + str((dpoint+1)*nb_epoch_per_dpoint) + ' of rerun ' + str(rerun+1))
                print("*" * 40)

        acti_train[:,:,rerun] = model_truncated.predict(X_train, batch_size=batch_size)
        acti_test[:,:,rerun] = model_truncated.predict(X_test, batch_size=batch_size)

        end = time.time()

        print("*" * 40)
        print('Estimated remaining run time: ' + str(int((end-start)*(nb_reruns-(rerun+1)))) + ' sec')
        print("*" * 40)

    return metrics, (acti_train, acti_test), model if return_model else None


def fit_pseudo(self, X_train, nb_pseudos, batch_size, nb_epoch,
               get_pseudos_func, train=True, validation_data=None,
               train_on_only_original=False, test_on_only_original=True):

    if validation_data is not None:
        print('\nTrain on %d samples, validate on %d samples' %
              (len(X_train)*nb_pseudos, len(validation_data[0])))
    else:
        print('\nTrain on %d samples' % (len(X_train)*nb_pseudos))

    progbar = generic_utils.Progbar(nb_epoch)

    for epoch in range(nb_epoch):

        if train:
            #train model
            for X_batch, Y_batch in pseudo_batch_generator(X_train, batch_size, nb_pseudos, get_pseudos_func, train_on_only_original):
                self.train_on_batch(X_batch, Y_batch)

        #test on original training set
        count = 0
        history_train = np.zeros(2)

        for X_batch, Y_batch in pseudo_batch_generator(X_train, batch_size, nb_pseudos, get_pseudos_func, validate_on_only_original):
            history_train += self.test_on_batch(X_batch, Y_batch)
            count += 1

        history_train /= count
        history_train = list(history_train)
        values=[('loss', history_train[0]), ('acc', history_train[1])]

        #test on original test set if validate_on_test_set is True
        if validation_data is not None:

            count = 0
            history_test = np.zeros(2)

            for X_batch, Y_batch in pseudo_batch_generator(validation_data[0], batch_size, nb_pseudos, get_pseudos_func, validate_on_only_original):
                history_test += self.test_on_batch(X_batch, Y_batch)
                count += 1

            history_test /= count
            history_test = list(history_test)
            values.extend([('val_loss', history_test[0]), ('val_acc', history_test[1])])
            history_train.extend(history_test)

        progbar.add(1, values=values)


    return history_train


Model.fit_pseudo = fit_pseudo


def pseudo_batch_generator(X_train, batch_size, nb_pseudos, get_pseudos_func, only_original):

    if only_original:
        #create a suffled index equal to original dataset size
        ind = np.random.permutation(len(X_train))
    else:
        #create a suffled index equal to original dataset size * nb_pseudos
        ind = np.random.permutation(nb_pseudos*len(X_train))

    for ind_batch_start in range(0, len(ind), batch_size):

        #if the last batch then the size is the number of remaining samples
        if ind_batch_start+batch_size > len(ind):
            ind_batch = ind[ind_batch_start::]
        else:
            ind_batch = ind[ind_batch_start:ind_batch_start+batch_size]

        X_batch = X_train[ind_batch%len(X_train),]

        if only_original:
            #create all zeros output labels
            y_batch = np.zeros_like(ind_batch)
        else:
            #create suffled eqaully distributed labels with values 0 to nb_pseudos
            y_batch = ind_batch/len(X_train)

        #in case if nb_pseudos=1 to support null_node
        if nb_pseudos > 1:
            Y_batch = np_utils.to_categorical(y_batch, nb_pseudos)
        else:
            Y_batch = np_utils.to_categorical(y_batch, nb_pseudos+1)

        for pseudo in range(nb_pseudos):
            X_batch[y_batch==pseudo,] = get_pseudos_func(X_batch[y_batch==pseudo,], pseudo)


        yield X_batch, Y_batch


def evaluate_clustering(self, X, y, nb_all_clusters, batch_size=128, cluster_offset=0, label_correction=False):

    #calculate clustering accuracy
    est = self.predict_classes(X, batch_size=batch_size, verbose=0)
    cl_acc = calculate_cl_acc(y, est, nb_all_clusters, cluster_offset, label_correction)

    #Check if clustering accuracy is calculated over entire dataset
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


def initialize_training_variables(nb_pseudos, nb_clusters_per_pseudo,
                                  y_train, y_test,
                                  nb_reruns, nb_epoch, nb_dpoints):

    #find the values of the dependent variables used inside the script
    nb_all_clusters = nb_pseudos*nb_clusters_per_pseudo
    nb_classes = y_train.max() - y_train.min() + 1
    nb_epoch_per_dpoint = nb_epoch/nb_dpoints

    # y to Y conversion for original dataset
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

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

    #initialize activation matrices
    acti_train = np.zeros((len(y_train), nb_all_clusters, nb_reruns))
    acti_test = np.zeros((len(y_test), nb_all_clusters, nb_reruns))

    return metrics, (Y_train, Y_test), (acti_train, acti_test), nb_all_clusters, nb_classes, nb_epoch_per_dpoint
