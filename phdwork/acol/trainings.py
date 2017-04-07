import numpy as np
import time
from keras import backend as K
from keras.utils import np_utils
from phdwork.commons.utils import calculate_cl_acc, cumulate_metrics

'''

Functions to train and evaluate ACOL experiments.

'''

def train_acol_models_for_parentvised(nb_parents, nb_clusters_per_parent, model, model_truncated,
                                      X_train, y_train, y_train_parent,
                                      X_test, y_test, y_test_parent,
                                      optimizer, nb_reruns, nb_epoch, nb_dpoints, batch_size,
                                      validate_on_test_set=True, c3_update_func=None):

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

        #compile models for each rerun
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
                model.get_layer("L-1").activity_regularizer.c3.set_value(c3_update_func(acol_metrics))
                model_truncated.get_layer("L-1").activity_regularizer.c3.set_value(c3_update_func(acol_metrics))

            metrics.get('loss')[-1].append(history.history.get('loss')[0])
            metrics.get('acc')[-1].append(history.history.get('acc')[0])
            metrics.get('vloss')[-1].append(history.history.get('val_loss')[0])
            metrics.get('vacc')[-1].append(history.history.get('val_acc')[0])

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

    return metrics, (acti_train, acti_test)
