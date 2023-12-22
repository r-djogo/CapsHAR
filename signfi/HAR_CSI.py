"""
HAR Using CSI WiFi - Data and ML Modeling Helper Functions
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
import tensorflow as tf
import time
from keras import Model, layers
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.regularizers import l2
from progressbar import Bar, Percentage, ProgressBar, Timer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from utils.layers import FCCaps

from utils.layers import FCCaps, Length, PrimaryCaps, Mask
from utils.tools import marginLoss

### Data Getting Functions

def get_csi(directory, subjects, APs, orientations, gestures, include_nomove=False):
    """
    Loads all the CSI data specified by the parameters.

    Parameters
    ----------
    directory: str
        Head directory containing all the CSI data
        '/home/...'
    subjects: list
        Which subjects to include in data
        ['subject1', ..., 'subject6']
    APs: list
        Which APs to include in data
        ['AP0', 'AP1', 'AP2', 'AP3', 'AP4']
    orientations: list
        Which orientations to include in data
        ['0', '45', '90', '180']
    gestures: list
        Which gestures to include in data
        ['circle', 'leftright', 'updown', 'pushpull']
    include_nomove: (optional) bool
        Whether or not to include nomove data
    
    Returns
    -------
    x: list
        x values of dataset
    y: list
        y values of dataset
    """
    
    x = []
    y = []
    max = len(APs)*len(gestures)*len(orientations)*len(subjects)
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=max).start()

    for i, subj in enumerate(subjects):
        for j, orient in enumerate(orientations):
            for k, gest in enumerate(gestures):
                gesture_file_order = []

                for l, ap in enumerate(APs):
                    # generate index for the specific gesture, person, and orientation
                    index = 1 + l + \
                            k*len(APs) + \
                            j*len(APs)*len(gestures) + \
                            i*len(APs)*len(gestures)*len(orientations)

                    # ensure that the order of gesture files in directory remains consistant ...
                    # so that the combine_ap_data() function can work properly
                    dir_contents = sorted(os.listdir(directory + subj + "/" + ap + "/csi/" + orient + "/" + gest))

                    #### temporary fix for missing file 'nomove_20.mat'
                    if subj == 'subject5' and orient == '0' and gest == 'pushpull' and ap != 'AP1':
                        dir_contents.remove('nomove_20.mat')

                    if l == 0:
                        gesture_file_order = dir_contents
                    else:
                        if gesture_file_order != dir_contents:
                            print("ERROR: gesture_file_order does not match")
                            print(gesture_file_order, dir_contents)
                            exit()

                    # load each gesture file in the directory
                    for file in gesture_file_order:
                        # load .mat file
                        if file.startswith('action'):
                            csi = scipy.io.loadmat(directory + subj + "/" + ap + "/csi/" +
                                                   orient + "/" + gest + "/" + file)['action_signal']
                        elif file.startswith('nomove'):
                            csi = scipy.io.loadmat(directory + subj + "/" + ap + "/csi/" +
                                                   orient + "/" + gest + "/" + file)['nomove_signal']
                        
                        # do any processing necessary to the CSI data here
                        # extract active subcarriers and correct length
                        # active_sc = list(range(6,32)) + list(range(33,59)) + list(range(70,96)) + \
                        #             list(range(97,123)) + list(range(134,160)) + list(range(161,187))
                        # active_sc = list(range(25,32)) + list(range(89,96)) + list(range(153,160))
                        # csi = csi[active_sc, 0:300]

                        # csi = np.concatenate((np.abs(csi), np.angle(csi)), axis=0)
                        # csi = np.stack((np.abs(csi), np.angle(csi)), axis=0)
                        # csi = np.concatenate((np.abs(csi), np.diff(np.angle(csi), axis=0)), axis=0)
                        csi = np.abs(csi)

                        # plt.plot(csi[0,:])
                        # apply moving average filter to csi magnitudes along rows
                        # filter_len = 5
                        # for t_step in range(csi.shape[1]):
                        #     csi[:, t_step] = np.mean(csi[:, t_step:t_step+filter_len], axis=1)
                        # plt.plot(csi[0,:])
                        # plt.savefig('plotting.png', bbox_inches='tight')
                        # plt.close()
                        # exit()

                        csi = (csi - np.mean(csi, axis=1)[:, np.newaxis]) / np.std(csi, axis=1)[:, np.newaxis]

                        # need axis 0 to be time steps
                        csi = np.transpose(csi)

                        # append to dataset
                        if file.startswith('action'):
                            # append to data
                            x.append(np.array(csi))
                            # append index as label
                            y.append(index)
                        elif file.startswith('nomove') and include_nomove:
                            # append to data
                            x.append(np.array(csi))
                            # append -index as label for nomove
                            y.append(-index)
                    pbar.update(index)
    
    pbar.finish()
    return x, y

def SFFT(x):
    # % 1. transpose the sequence x to convert rows into columns
    # % 2. execute fft() command to compute fft of the columns (which are originally rows)
    # % 3. transpose the matrix back to original form
    # % 4. execute ifft on the columns
    return np.fft.fftshift(np.fft.ifft(np.transpose(np.fft.fft(np.transpose(x))))) # fftshift(ifft(fft(x.').'))

def get_csi_SFFT(directory, subjects, APs, orientations, gestures, include_nomove=False):
    """
    Loads all the CSI data specified by the parameters.

    Parameters
    ----------
    directory: str
        Head directory containing all the CSI data
        '/home/...'
    subjects: list
        Which subjects to include in data
        ['subject1', ..., 'subject6']
    APs: list
        Which APs to include in data
        ['AP0', 'AP1', 'AP2', 'AP3', 'AP4']
    orientations: list
        Which orientations to include in data
        ['0', '45', '90', '180']
    gestures: list
        Which gestures to include in data
        ['circle', 'leftright', 'updown', 'pushpull']
    include_nomove: (optional) bool
        Whether or not to include nomove data
    
    Returns
    -------
    x: list
        x values of dataset
    y: list
        y values of dataset
    """
    
    x = []
    y = []
    max = len(APs)*len(gestures)*len(orientations)*len(subjects)
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=max).start()

    for i, subj in enumerate(subjects):
        for j, orient in enumerate(orientations):
            for k, gest in enumerate(gestures):
                gesture_file_order = []

                for l, ap in enumerate(APs):
                    # generate index for the specific gesture, person, and orientation
                    index = 1 + l + \
                            k*len(APs) + \
                            j*len(APs)*len(gestures) + \
                            i*len(APs)*len(gestures)*len(orientations)

                    # ensure that the order of gesture files in directory remains consistant ...
                    # so that the combine_ap_data() function can work properly
                    dir_contents = sorted(os.listdir(directory + subj + "/" + ap + "/csi/" + orient + "/" + gest))

                    #### temporary fix for missing file 'nomove_20.mat'
                    if subj == 'subject5' and orient == '0' and gest == 'pushpull' and ap != 'AP1':
                        dir_contents.remove('nomove_20.mat')
                    #### temporary fix for (479,192) file 'nomove_16.mat'
                    if subj == 'subject3' and orient == '90' and gest == 'pushpull':
                        dir_contents.remove('nomove_16.mat')

                    if l == 0:
                        gesture_file_order = dir_contents
                    else:
                        if gesture_file_order != dir_contents:
                            print("ERROR: gesture_file_order does not match")
                            print(gesture_file_order, dir_contents)
                            exit()

                    # load each gesture file in the directory
                    for file in gesture_file_order:
                        # load .mat file
                        if file.startswith('action'):
                            csi = scipy.io.loadmat(directory + subj + "/" + ap + "/csi/" +
                                                   orient + "/" + gest + "/" + file)['action_signal']
                        elif file.startswith('nomove'):
                            csi = scipy.io.loadmat(directory + subj + "/" + ap + "/csi/" +
                                                   orient + "/" + gest + "/" + file)['nomove_signal']
                        
                        ### do any processing necessary to the CSI data here
                        csi = np.transpose(csi)

                        # take SFFT in 16 sample intervals
                        for step in range(0, csi.shape[0]-4, 16): # ends at 496 so that all windows at size (16,192)
                            sfft_csi_step = SFFT(csi[step:step+16, :])
                            if step == 0:
                                sfft_csi = sfft_csi_step[np.newaxis]
                            else:
                                sfft_csi = np.concatenate((sfft_csi, sfft_csi_step[np.newaxis]), axis=0)
                        
                        # add channels as last axis to the data
                        sfft_csi = sfft_csi[..., np.newaxis]
                        sfft_csi_mag = np.abs(sfft_csi)
                        sfft_csi_mag = (sfft_csi_mag - np.mean(sfft_csi_mag)) / np.std(sfft_csi_mag)
                        # sfft_csi_ang = np.angle(sfft_csi)
                        # sfft_csi_ang = (sfft_csi_ang - np.mean(sfft_csi_ang)) / np.std(sfft_csi_ang)
                        # sfft_csi = np.concatenate((sfft_csi_mag, sfft_csi_ang), axis=3)
                        sfft_csi = sfft_csi_mag

                        # append to dataset
                        if file.startswith('action'):
                            # append to data
                            x.append(np.array(sfft_csi))
                            # append index as label
                            y.append(index)
                        elif file.startswith('nomove') and include_nomove:
                            # append to data
                            x.append(np.array(sfft_csi))
                            # append -index as label for nomove
                            y.append(-index)
                    pbar.update(index)
    
    pbar.finish()
    return x, y

def unison_shuffled_copies(a, b):
    """
    Shuffles both input arrays in unison.
    """
    
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def combine_ap_data(data, labels, n_APs):
    """
    Combines data from different APs into single matrix.

    Parameters
    ----------
    data: list
        data values of dataset
    labels: list
        label values of dataset
    n_APs: int
        total number of APs in dataset
    
    Returns
    -------
    x: list
        data values of dataset with APs combined
    y: list
        label values of dataset with APs combined
    """
    
    x = []
    y = []

    # handle gestures (label > 0)
    for i in range(1, max(labels)+1, n_APs):
        idxs = list(range(i, i+n_APs))
        AP_data = []
        for idx in idxs:
            first = labels.index(idx)
            last = len(labels) - 1 - labels[::-1].index(idx)
            AP_data.append(data[first:last+1])

        for j in range(1,len(AP_data)):
            for k in range(len(AP_data[0])):
                AP_data[0][k] = np.concatenate([AP_data[0][k], AP_data[j][k]], axis=1)
                # AP_data[0][k] = np.dstack((AP_data[0][k], AP_data[j][k]))

        x.extend(np.array(AP_data[0]))
        y.extend([i] * len(AP_data[0]))

    # handle nomove cases (label < 0)
    neg_labels = [-a for a in labels]
    for i in range(1, max(neg_labels)+1, n_APs):
        idxs = list(range(i, i+n_APs))
        AP_data = []
        for idx in idxs:
            first = neg_labels.index(idx)
            last = len(neg_labels) - 1 - neg_labels[::-1].index(idx)
            if (last-first) > 5:
                last = first + 4
            AP_data.append(data[first:last+1])

        for j in range(1,len(AP_data)):
            for k in range(len(AP_data[0])):
                AP_data[0][k] = np.concatenate([AP_data[0][k], AP_data[j][k]], axis=1)
                # AP_data[0][k] = np.dstack((AP_data[0][k], AP_data[j][k]))

        x.extend(np.array(AP_data[0]))
        y.extend([0] * len(AP_data[0]))
        
    return x, y

def combine_ap_data_SFFT(data, labels, n_APs):
    """
    Combines data from different APs into single matrix.

    Parameters
    ----------
    data: list
        data values of dataset
    labels: list
        label values of dataset
    n_APs: int
        total number of APs in dataset
    
    Returns
    -------
    x: list
        data values of dataset with APs combined
    y: list
        label values of dataset with APs combined
    """
    
    x = []
    y = []

    # handle gestures (label > 0)
    for i in range(1, max(labels)+1, n_APs):
        idxs = list(range(i, i+n_APs))
        AP_data = []
        for idx in idxs:
            first = labels.index(idx)
            last = len(labels) - 1 - labels[::-1].index(idx)
            AP_data.append(data[first:last+1])

        for j in range(1,len(AP_data)):
            for k in range(len(AP_data[0])):
                AP_data[0][k] = np.concatenate([AP_data[0][k], AP_data[j][k]], axis=3)

        x.extend(np.array(AP_data[0]))
        y.extend([i] * len(AP_data[0]))

    # handle nomove cases (label < 0)
    neg_labels = [-a for a in labels]
    for i in range(1, max(neg_labels)+1, n_APs):
        idxs = list(range(i, i+n_APs))
        AP_data = []
        for idx in idxs:
            first = neg_labels.index(idx)
            last = len(neg_labels) - 1 - neg_labels[::-1].index(idx)
            if (last-first) > 5:
                last = first + 4
            AP_data.append(data[first:last+1])

        for j in range(1,len(AP_data)):
            for k in range(len(AP_data[0])):
                AP_data[0][k] = np.concatenate([AP_data[0][k], AP_data[j][k]], axis=3)

        x.extend(np.array(AP_data[0]))
        y.extend([0] * len(AP_data[0]))
        
    return x, y

def group_data(labels, groupings):
    """
    Groups different labels according to the parameters given.

    Parameters
    ----------
    labels: list
        label values of dataset
    groupings: list
        list of the label groupings to be combined
    
    Returns
    -------
    labels: list
        new label values of dataset
    """

    for group in groupings:
        for i in range(len(labels)):
            if labels[i] in group:
                labels[i] = group[0]

    return labels

def simplify_labels(labels):
    """
    Simplifies labels such that they are the set of the smallest possible integers.

    Parameters
    ----------
    labels: list
        label values of dataset
    
    Returns
    -------
    labels: list
        new label values of dataset
    """

    unique_labels = np.unique(labels)
    new_unique_labels = {}
    for n, label in enumerate(unique_labels):
        new_unique_labels[label] = n

    for i in range(len(labels)):
        if labels[i] < 0:
            labels[i] = 0
        else:
            labels[i] = new_unique_labels[labels[i]]

    return labels

def normalize_data(x, y):
    """
    Normalizes the data and converts labels to One-Hot encoding.

    Parameters
    ----------
    x: list
        data values of dataset
    y: list
        label values of dataset
    
    Returns
    -------
    x: list
        data values of dataset normalized
    y: list
        label values of dataset as One-Hot vectors
    """

    # normalize
    x = np.array(x)
    x = (x - np.mean(x)) / np.std(x)
    y = np.array(y).reshape(-1,1)

    # encode classes
    label_encoder = OneHotEncoder(sparse=False)
    y = label_encoder.fit_transform(y)

    return x, y

def shuffle_split_data(x, y, split=0.7):
    """
    Shuffles and splits the data.

    Parameters
    ----------
    x: list
        data values of dataset
    y: list
        label values of dataset
    split: (optional) Number in range [0,1]
        ratio of data to be put in training set
    
    Returns
    -------
    x_train: list
        data values of training dataset shuffled
    y_train: list
        label values of training dataset shuffled
    x_test: list
        data values of testing dataset shuffled
    y_test: list
        label values of testing dataset shuffled
    """

    # shuffle data
    x_s, y_s = unison_shuffled_copies(x,y)

    # shuffle and split data
    if split < 1.0:
        x_train, x_test, y_train, y_test = train_test_split(x_s, y_s, train_size=split, shuffle=True, stratify=y_s)
        # x_train, y_train = x_s[0:int(split*len(x_s)),:,:], y_s[0:int(split*len(y_s)),:]
        # x_test, y_test = x_s[int(split*len(x_s)):,:,:], y_s[int(split*len(y_s)):,:]
        return x_train, y_train, x_test, y_test
    else:
        return x_s, y_s, [], []

### ML Modeling Functions

def SqueezeAndExcitation(inputs, ratio=16):
    b, _, _, ch = inputs.shape
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(ch//ratio, activation='relu')(x)
    x = tf.keras.layers.Dense(ch, activation='sigmoid')(x)
    return tf.keras.layers.multiply([inputs, x])

def data_to_tensor(x_train, y_train, x_test, y_test):
    x_train = tf.convert_to_tensor(np.array(x_train), dtype=tf.float32)
    y_train = tf.convert_to_tensor(np.array(y_train), dtype=tf.float32)
    x_test = tf.convert_to_tensor(np.array(x_test), dtype=tf.float32)
    y_test = tf.convert_to_tensor(np.array(y_test), dtype=tf.float32)
    return x_train, y_train, x_test, y_test

def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.0001)

def train_model(model, lr, batch_size, epochs, x_train, y_train, x_test, y_test, verbose=1):
    opt = keras.optimizers.Adam(learning_rate=lr)
    # opt = keras.optimizers.SGD(learning_rate=lr)
    mcp_save = ModelCheckpoint('best.mdl_HAR.hdf5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['categorical_accuracy'])
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size,
                        validation_data=(x_test, y_test), callbacks=[mcp_save, lr_callback], shuffle=True)

    return history

def plot_model(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    # plt.show()
    plt.savefig('./results/mdl_accuracy.png', bbox_inches='tight')
    plt.close()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig('./results/mdl_loss.png', bbox_inches='tight')
    plt.close()

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch >= 40:  # or save after some epoch, each k-th epoch etc.
            self.model.save(f"./results/model_{epoch}.hdf5")

# make an ensemble prediction for multi-class classification
def load_ensemble_predictions(x_test, name, e_start=40, e_end=50):
    all_models = list()
    for epoch in range(e_start, e_end):
        # define filename for this ensemble
        filename = name + str(epoch) + '.hdf5'
        # load model from file
        model = keras.models.load_model(filename, custom_objects={"PrimaryCaps": PrimaryCaps, "marginLoss": marginLoss,
                                        "FCCaps": FCCaps, "Length": Length})
        # add to list of members
        all_models.append(model)
    # make predictions
    yhats = [model.predict(x_test) for model in all_models]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = summed / 10
    return result

def load_model_predictions(x_test, batch_size, name="best.mdl_HAR.hdf5"):
    model = keras.models.load_model(name, custom_objects={"PrimaryCaps": PrimaryCaps, "marginLoss": marginLoss,
                                                          "FCCaps": FCCaps, "Length": Length})
    
    t_start = time.time()
    predictions = model.predict(x_test, verbose=0)#, batch_size=batch_size)
    inference_time = time.time() - t_start

    return predictions, inference_time

def load_model_predictions_generator(x_test, batch_size, model, name="best.mdl_HAR.hdf5"):
    # model = keras.models.load_model(name, custom_objects={"PrimaryCaps": PrimaryCaps, "marginLoss": marginLoss,
    #                                                       "FCCaps": FCCaps, "Length": Length, "Mask": Mask})
    
    t_start = time.time()
    predictions, x_recon = model.predict(x_test, verbose=0)#, batch_size=batch_size)
    # predictions = model.predict(x_test, verbose=0)#, batch_size=batch_size)
    inference_time = time.time() - t_start

    return predictions, inference_time

def create_conf_matrix(y_test, predictions):
    matrix = confusion_matrix(np.argmax(y_test,1), np.argmax(predictions,1), normalize='true').astype(float)
    # sums = matrix.sum(axis=1)
    # for i in range(2):
    #   matrix[i,:]=matrix[i,:] / sums[i]
    matrix = np.round(matrix,3) * 100

    return matrix

def plot_conf_matrix_2(matrix, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix %' + title)
    ax.xaxis.set_ticklabels(['circle', 'leftright'])
    ax.yaxis.set_ticklabels(['circle', 'leftright'])
    # plt.show()
    plt.savefig('mdl_conf_matrix.png', bbox_inches='tight')
    plt.close()

def plot_conf_matrix_3(matrix, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix %' + title)
    ax.xaxis.set_ticklabels(['none', 'circle', 'leftright'])
    ax.yaxis.set_ticklabels(['none', 'circle', 'leftright'])
    # plt.show()
    plt.savefig('mdl_conf_matrix.png', bbox_inches='tight')
    plt.close()

def plot_conf_matrix_4(matrix, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax);
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)
    # ax.xaxis.set_ticklabels(['0', '45', '90', '180'])
    # ax.yaxis.set_ticklabels(['0', '45', '90', '180'])
    ax.xaxis.set_ticklabels(['circle', 'leftright', 'updown', 'pushpull'])
    ax.yaxis.set_ticklabels(['circle', 'leftright', 'updown', 'pushpull'])
    # plt.show()
    plt.savefig('mdl_conf_matrix.png', bbox_inches='tight')
    plt.close()

def plot_conf_matrix_5(matrix, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)
    # ax.xaxis.set_ticklabels(['run_1', 'run_2', 'run_3', 'run_4', 'run_5'])
    # ax.yaxis.set_ticklabels(['run_1', 'run_2', 'run_3', 'run_4', 'run_5'])

    ax.xaxis.set_ticklabels(['none', 'circle', 'leftright', 'updown', 'pushpull'])
    ax.yaxis.set_ticklabels(['none', 'circle', 'leftright', 'updown', 'pushpull'])
    # plt.show()
    plt.savefig('./results/mdl_conf_matrix.png', bbox_inches='tight')
    plt.close()

def plot_conf_matrix_6(matrix, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)

    ax.xaxis.set_ticklabels(['bed', 'fall', 'walk', 'run', 'sitdown', 'standup'])
    ax.yaxis.set_ticklabels(['bed', 'fall', 'walk', 'run', 'sitdown', 'standup'])
    plt.savefig('./results/mdl_conf_matrix.png', bbox_inches='tight')
    plt.close()

def plot_conf_matrix_7(matrix, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)

    ax.xaxis.set_ticklabels(['bed', 'fall', 'walk', 'pickup', 'run', 'sitdown', 'standup'])
    ax.yaxis.set_ticklabels(['bed', 'fall', 'walk', 'pickup', 'run', 'sitdown', 'standup'])
    plt.savefig('./results/mdl_conf_matrix.png', bbox_inches='tight')
    plt.close()

def plot_conf_matrix_6_file(matrix, file, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)

    ax.xaxis.set_ticklabels(['bed', 'fall', 'walk', 'run', 'sitdown', 'standup'])
    ax.yaxis.set_ticklabels(['bed', 'fall', 'walk', 'run', 'sitdown', 'standup'])
    plt.savefig(file, bbox_inches='tight')
    plt.close()

def plot_conf_matrix_7_file(matrix, file, title=''):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)

    ax.xaxis.set_ticklabels(['bed', 'fall', 'walk', 'pickup', 'run', 'sitdown', 'standup'])
    ax.yaxis.set_ticklabels(['bed', 'fall', 'walk', 'pickup', 'run', 'sitdown', 'standup'])
    plt.savefig(file, bbox_inches='tight')
    plt.close()

def plot_conf_matrix_17(matrix, title=''):
    fig=plt.figure(figsize=(12,10))

    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax); 
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)

    ax.xaxis.set_ticklabels(['none', 'circle-0', 'leftright-0', 'updown-0', 'pushpull-0',
                                     'circle-45', 'leftright-45', 'updown-45', 'pushpull-45',
                                     'circle-90', 'leftright-90', 'updown-90', 'pushpull-90',
                                     'circle-180', 'leftright-180', 'updown-180', 'pushpull-180'])
    plt.xticks(rotation=90) 
    ax.yaxis.set_ticklabels(['none', 'circle-0', 'leftright-0', 'updown-0', 'pushpull-0',
                                     'circle-45', 'leftright-45', 'updown-45', 'pushpull-45',
                                     'circle-90', 'leftright-90', 'updown-90', 'pushpull-90',
                                     'circle-180', 'leftright-180', 'updown-180', 'pushpull-180'])
    plt.yticks(rotation=0) 

    # plt.show()
    plt.savefig('mdl_conf_matrix_17.png', bbox_inches='tight')
    plt.close()