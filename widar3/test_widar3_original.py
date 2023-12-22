from __future__ import print_function

import os,sys
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, TimeDistributed
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix
# from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
import gc
import time
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score
from HAR_CSI import *


# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = '/home/...'
ALL_MOTION = [1,2,3,4,5,6,7,8,9,10]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
n_epochs = 200
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 32
f_learning_rate = 0.001
n_runs = 10


# Initializing the performance variables
matrices = []
training_times = []
inference_times = []
len_train = 0
len_test = 0
hists = []
MCC = []
F1 = []
B_ACC = []
ACC = []


def normalize_data(data_1):
    # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
    data_1_max = np.concatenate((data_1.max(axis=0),data_1.max(axis=1)),axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0),data_1.min(axis=1)),axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max,(data_1.shape[0],data_1.shape[1],1))
    data_1_min_rep = np.tile(data_1_min,(data_1.shape[0],data_1.shape[1],1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return  data_1_norm

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())
    return np.array(data_pad)

def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     # from label to onehot
    return _label

def load_data(path_to_data, motion_sel):
    global T_MAX
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):

        pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=len(data_files)).start()

        for i, data_file_name in enumerate(data_files):

            file_path = os.path.join(data_root,data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                location = int(data_file_name.split('-')[2])
                orientation = int(data_file_name.split('-')[3])
                repetition = int(data_file_name.split('-')[4])

                # Select Motion
                if (label_1 not in motion_sel):
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue
                
                # Normalization
                data_normed_1 = normalize_data(data_1)
                
                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]                
            except Exception:
                continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1)

            pbar.update(i+1)
        pbar.finish()
            
    # Zero-padding
    data = zero_padding(data, T_MAX)

    # Swap axes
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
    data = np.expand_dims(data, axis=-1)    # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1]

    # Convert label to ndarray
    label = np.array(label)

    # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
    return data, label
    
def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')    # (@,T_MAX,20,20,1)

    # Feature extraction part
    x = TimeDistributed(Conv2D(16,kernel_size=(5,5),activation='relu',data_format='channels_last',\
        input_shape=input_shape))(model_input)   # (@,T_MAX,20,20,1)=>(@,T_MAX,16,16,16)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)    # (@,T_MAX,16,16,16)=>(@,T_MAX,8,8,16)
    x = TimeDistributed(Flatten())(x)   # (@,T_MAX,8,8,16)=>(@,T_MAX,8*8*16)
    x = TimeDistributed(Dense(64,activation='relu'))(x) # (@,T_MAX,8*8*16)=>(@,T_MAX,64)
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)
    x = TimeDistributed(Dense(64,activation='relu'))(x) # (@,T_MAX,64)=>(@,T_MAX,64)
    x = GRU(n_gru_hidden_units,return_sequences=False)(x)  # (@,T_MAX,64)=>(@,128)
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)  # (@,128)=>(@,n_class)

    # Model compiling
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
    return model

# Load data
data, label = load_data(data_dir, ALL_MOTION)
print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0,:,:].shape) + '\n')

for run in range(n_runs):

    # Split train and test
    [data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
    print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +\
        'Test on ' + str(label_test.shape[0]) + ' samples\n')
    len_train = label_train.shape[0] * 0.9
    len_test = label_test.shape[0]

    # One-hot encoding for train data
    label_train = onehot_encoding(label_train, N_MOTION)
    label_test = onehot_encoding(label_test, N_MOTION)


    # Load or fabricate model
    if use_existing_model:
        model = load_model('model_widar3_trained.h5')
        model.summary()
    else:
        model = assemble_model(input_shape=(T_MAX, 20, 20, 1), n_class=N_MOTION)
        if run == 0:
            model.summary()
        t_start = time.time()
        history = model.fit({'name_model_input': data_train},{'name_model_output': label_train},
                batch_size=n_batch_size,
                epochs=n_epochs,
                verbose=0,
                validation_split=0.1, shuffle=True)
        training_times.append((time.time() - t_start))
        hists.append(history)
        print('Saving trained model...')
        model.save('./results/final_model_widar.hdf5')
        pred, inference_time = load_model_predictions(data_test, n_batch_size, name='./results/final_model_widar.hdf5')
        inference_times.append(inference_time)

        matrices.append(create_conf_matrix(label_test, pred))
        MCC.append(matthews_corrcoef(np.argmax(label_test, axis=1), np.argmax(pred, axis=1)))
        F1.append(f1_score(np.argmax(label_test, axis=1), np.argmax(pred, axis=1), average='micro'))
        B_ACC.append(balanced_accuracy_score(np.argmax(label_test, axis=1), np.argmax(pred, axis=1)))
        ACC.append(accuracy_score(np.argmax(label_test, axis=1), np.argmax(pred, axis=1)))

        del data_train, label_train, data_test, label_test, model
        gc.collect()

# Write results to file
f_res = open("./results/matrices.txt", "w")
f_res.write(str(matrices))
all_matrices = np.array(matrices)
avg_matrix = np.mean(all_matrices, axis=0)
std_matrix = np.std(all_matrices, axis=0)
var_matrix = np.var(all_matrices, axis=0)
f_res.write("\nAvg model matrix:\n")
f_res.write(str(avg_matrix))
f_res.write("\nStd model matrix:\n")
f_res.write(str(std_matrix))
f_res.write("\nVar model matrix:\n")
f_res.write(str(var_matrix))
f_res.write("\nMCC:")
f_res.write(str(MCC))
f_res.write("\nAvg MCC:")
f_res.write(str(np.mean(MCC)))
f_res.write("\nStd MCC:")
f_res.write(str(np.std(MCC)))
f_res.write("\nF1:")
f_res.write(str(F1))
f_res.write("\nAvg F1:")
f_res.write(str(np.mean(F1)))
f_res.write("\nStd F1:")
f_res.write(str(np.std(F1)))
f_res.write("\nBalanced accuracy:")
f_res.write(str(B_ACC))
f_res.write("\nAvg balanced accuracy:")
f_res.write(str(np.mean(B_ACC)))
f_res.write("\nStd balanced accuracy:")
f_res.write(str(np.std(B_ACC)))
f_res.write("\nAccuracy score:")
f_res.write(str(ACC))
f_res.write("\nAvg accuracy:")
f_res.write(str(np.mean(ACC)))
f_res.write("\nStd accuracy:")
f_res.write(str(np.std(ACC)))
f_res.write("\nAvg accuracy (from matrices):")
f_res.write(str(np.round(np.mean(avg_matrix.diagonal()), 2)))
f_res.write("\nTraining times:")
f_res.write(str(training_times))
f_res.write("\nAvg training time:")
f_res.write(str(np.mean(training_times)))
f_res.write("\nAvg training time per sample:")
f_res.write(str(np.mean(training_times) / n_epochs / len_train))
f_res.write("\nInference times:")
f_res.write(str(inference_times))
f_res.write("\nAvg inference time:")
f_res.write(str(np.mean(inference_times)))
f_res.write("\nAvg inference time per sample:")
f_res.write(str(np.mean(inference_times) / len_test))
f_res.write("\n")
for hist in hists:
    f_res.write("Single run results:\n")
    for value in hist.history:
        f_res.write(str(value))
        f_res.write(str(hist.history[value]))
        f_res.write("\n")
f_res.close()

if N_MOTION == 10:
    plot_conf_matrix_10(avg_matrix)
else:
    print(avg_matrix)