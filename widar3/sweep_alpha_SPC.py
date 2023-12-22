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
from utils.layers import FCCaps, Length, PrimaryCaps, Mask
from utils.tools import marginLoss


# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = '/home/...'
ALL_MOTION = [1,2,3,4,5,6,7,8,9,10]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
T_MIN = 99999
n_epochs = 100
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 16
lr = 5e-4
lr_decay = 0.98
n_runs = 10
verbose = 0

def normalize_data_widar(data_1):
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
    # data_pad = []
    # for i in range(len(data)):
    #     t = np.array(data[i]).shape[2]
    #     data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())
    # return np.array(data_pad)
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=len(data)).start()
    for i in range(len(data)):
        t = np.array(data[i]).shape[0]
        data[i] = np.pad(data[i], ((T_MAX - t,0),(0,0)), 'constant', constant_values = 0).astype(np.float32)
        pbar.update(i+1)
    pbar.finish()
    return np.array(data).astype(np.float32)

def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     # from label to onehot
    return _label

def load_data(path_to_data, motion_sel):
    global T_MAX, T_MIN
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        
        pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=len(data_files)).start()

        for i, data_file_name in enumerate(data_files):

            file_path = os.path.join(data_root,data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['doppler_spectrum']
                label_1 = int(data_file_name.split('-')[1])
                # location = int(data_file_name.split('-')[2])
                # orientation = int(data_file_name.split('-')[3])
                # repetition = int(data_file_name.split('-')[4])

                # Select Motion
                if (label_1 not in motion_sel):
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue
                
                data_1 = np.abs(data_1)
                # normalize
                for k in range(6):
                    data_1[k,:,:] = (data_1[k,:,:] - np.mean(data_1[k,:,:])) / np.std(data_1[k,:,:])
                # downsample by 5
                data_1 = data_1[:,::5,:]
                # reshape
                data_1 = np.hstack((data_1[0], data_1[1], data_1[2], data_1[3], data_1[4], data_1[5])).astype(np.float32) # (6,T,90) -> (T,540)

                # Normalization
                # data_normed_1 = normalize_data_widar(data_1)
                # # normalize each row separately
                # data_normed_1 = (data_1 - np.mean(data_1, axis=0)[np.newaxis,:]) / np.std(data_1, axis=0)[np.newaxis,:]
                # or basic:
                # data_normed_1 = (data_1 - np.mean(data_1)) / np.std(data_1)
                # print(data_normed_1.dtype)
                data_normed_1 = data_1
                
                # Update T_MAX
                if T_MAX < np.array(data_normed_1).shape[0]:
                    T_MAX = np.array(data_normed_1).shape[0]
                if T_MIN > np.array(data_normed_1).shape[0]:
                    T_MIN = np.array(data_normed_1).shape[0]
            except Exception:
                print("error")
                continue

            # Save List
            data.append(data_normed_1)
            label.append(label_1)

            pbar.update(i+1)
        pbar.finish()
            
    # Zero-padding
    data = zero_padding(data, T_MAX)

    # Swap axes
    # data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
    # data = np.expand_dims(data, axis=-1)    # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1]

    # Convert label to ndarray
    label = np.array(label)

    # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
    return data, label

def assemble_caps(input_shape, n_class):
    # Feature extraction part
    input = tf.keras.Input(shape=input_shape, dtype='float32')

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=16, strides=4, activation="relu",
                                padding='valid', kernel_regularizer='l2', bias_regularizer='l2')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=2, activation='relu',
                                padding='valid', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    pc = PrimaryCaps(F=16, K=8, N=1680, D=8, s=2)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(pc)
    activity_caps = FCCaps(n_class,16)(x)
    output = Length(name='length_capsnet_output')(activity_caps)
    capsnet = Model(inputs=[input], outputs=[pc, activity_caps, output])

    # generator graph using sub pixel convolution
    input_gen = tf.keras.Input(16*n_class, dtype='float32')
    x_gen = tf.keras.layers.Dense(units=29*27*1, activation="relu")(input_gen)
    x_gen = tf.keras.layers.Reshape(target_shape=(29, 27, 1))(x_gen)
    R = 20
    x_gen = tf.keras.layers.Conv2D(filters=R**2, kernel_size=5, padding='same')(x_gen)
    x_gen = tf.nn.depth_to_space(x_gen, R) # 580x540x1 output
    x_gen = tf.keras.layers.Cropping2D(cropping=((2, 3), (0, 0)))(x_gen) # 575x540x1 output
    generator = Model(inputs=input_gen, outputs=x_gen, name='Generator')

    # define combined models
    inputs = tf.keras.Input(shape=input_shape, dtype='float32')
    y_true = tf.keras.Input(shape=(n_class))
    prim_caps, out_caps, out_caps_len = capsnet(inputs)
    masked_by_y = Mask()([out_caps, y_true])  
    masked = Mask()(out_caps)

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)

    model = tf.keras.models.Model([inputs, y_true], [out_caps_len, x_gen_train], name='capsnet')
    model_test = tf.keras.models.Model(inputs, [out_caps_len, x_gen_eval], name='capsnet_test')

    return model, model_test, capsnet


# Load data
data, label = load_data(data_dir, ALL_MOTION)
print(T_MAX, T_MIN)

# print("loading")
# data = np.load('file.npy') # load
# label = np.load('file.npy') # load
# print("loaded")

print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0,:,:].shape) + '\n')

# test alpha values
alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] # centered around 4x

tf.compat.v1.disable_eager_execution()

for a_idx, alpha in enumerate(alphas):
    matrices = [] # variable for saving the confusion matrices
    training_times = [] # variable for saving training times
    inference_times = [] # variables for saving inference times
    len_train = 0
    len_test = 0
    hists = [] # variable for saving training histories
    MCC = []
    F1 = []
    B_ACC = []
    ACC = []
    val_B_ACC = []
    
    # make directory
    if not os.path.exists(f"./alpha_results/{str(alpha)}"):
        os.makedirs(f"./alpha_results/{str(alpha)}")

    print(f"Starting Model Training {a_idx+1}/{len(alphas)}")
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=n_runs).start()
    for run in range(n_runs):
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # tf.keras.backend.clear_session()

            # Split train and test
            [data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
            # print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +\
            #     'Test on ' + str(label_test.shape[0]) + ' samples\n')
            data_train = np.expand_dims(data_train, axis=3)
            data_test = np.expand_dims(data_test, axis=3)
            # print(data_train.shape, data_test.shape)
            len_train = label_train.shape[0] * 0.9
            len_test = label_test.shape[0]

            # One-hot encoding for train data
            label_train = onehot_encoding(label_train, N_MOTION)
            label_test = onehot_encoding(label_test, N_MOTION)

            # # model = assemble_model(input_shape=(T_MAX, 20, 20, 1), n_class=N_MOTION)
            model, model_test, capsnet = assemble_caps(input_shape=(data_train.shape[1], data_train.shape[2], data_train.shape[3]), n_class=N_MOTION)

            if run == 0 and a_idx == 0:
                # capsnet.summary()
                # generator.summary()
                model_test.summary()

            if os.path.isfile(f"./alpha_results/{str(alpha)}/mdl_HAR.hdf5"):
                os.remove(f"./alpha_results/{str(alpha)}/mdl_HAR.hdf5")

            # train the model
            adam = keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=adam, loss=[marginLoss, 'mse'], loss_weights=[alpha, (1-alpha)], metrics=['accuracy'])

            #callbacks
            log = tf.keras.callbacks.CSVLogger(f"./alpha_results/{str(alpha)}/log.csv", append=True)
            lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: max(lr * (lr_decay ** float(epoch)), 1e-4))
            callbacks = [log, lr_decay_cb]#, checkpoint]#, saver]

            #Train
            t_start = time.time()
            history = model.fit([data_train, label_train], [label_train, data_train], batch_size=n_batch_size, epochs=n_epochs,
                    validation_split=0.1, callbacks=callbacks, shuffle=True, verbose=verbose)
            training_times.append((time.time() - t_start))
            hists.append(history)

            try:
                val_B_ACC.append(history.history[list(history.history.keys())[-3]][-1])
            except:
                print("val_BAcc Didn't work...")

            model.save(f"./alpha_results/{str(alpha)}/final_model.hdf5")
            capsnet.save(f"./alpha_results/{str(alpha)}/final_model_capsnet.hdf5")

            del data_train, label_train
            gc.collect()

            pred, inference_time = load_model_predictions_generator(data_test, n_batch_size, model_test)
            inference_times.append(inference_time)

            # get performance metrics
            matrices.append(create_conf_matrix(label_test, pred))
            MCC.append(matthews_corrcoef(np.argmax(label_test, axis=1), np.argmax(pred, axis=1)))
            F1.append(f1_score(np.argmax(label_test, axis=1), np.argmax(pred, axis=1), average='micro'))
            B_ACC.append(balanced_accuracy_score(np.argmax(label_test, axis=1), np.argmax(pred, axis=1)))
            ACC.append(accuracy_score(np.argmax(label_test, axis=1), np.argmax(pred, axis=1)))

            print('Test acc:', [round(b,3) for b in B_ACC])
            print('Val acc:', [round(b,3) for b in val_B_ACC])

            # del data_train, label_train
            del data_test, label_test, adam, log, lr_decay_cb, callbacks, model, model_test, pred
            gc.collect()

            pbar.update(run+1)

    # Write results to file
    f_res = open(f"./alpha_results/{str(alpha)}/matrices.txt", "w")
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
        plot_conf_matrix_10_file(avg_matrix, f"./alpha_results/{str(alpha)}/conf_matrix.png")
    else:
        print(avg_matrix)

    pbar.finish()