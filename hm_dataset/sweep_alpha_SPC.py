import gc
import time
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score, mean_squared_error
from HAR_CSI import *
from utils.layers import FCCaps, Length, PrimaryCaps, Mask
from utils.tools import marginLoss

runs = 10 # number of iterations to perform
dir = '/home/...' # top level directory of data
subjects = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6'] # subjects to be included
# subjects = ['subject5'] # subjects to be included
APs = ['AP0','AP1','AP2','AP3','AP4'] # APs to be included
# APs = ['AP3', 'AP4'] # APs to be included
orients = ['0', '45', '90', '180'] # orientations to be included
# orients = ['180'] # orientations to be included
gestures = ['circle', 'leftright', 'updown', 'pushpull'] # gestures to be included
lr, lr_decay, batch_size, epochs = 5e-4, 0.95, 16, 100  # model training hyperparameters
verbose = 0 # 0 = silent, 1 = progress bar, 2 = one line per epoch
if runs > 1: verbose = 0

# group the data labels together in order to classify the appropriate category
include_nomove = True # whether to include nomove data
if include_nomove: offset = 1
else: offset = 0
total_groups = offset + len(subjects)*len(orients)*len(gestures) # doesn't include APs because we combine the APs into 1 AP
subject_groupings = [list(range(x,total_groups,int((total_groups-offset)/len(subjects)))) for x in range(offset,offset+int((total_groups-offset)/len(subjects)))] # groups together different subjects
# this one is for orientation classes # data_groupings = [list(range(x,x+len(gestures))) for x in range(offset,offset+len(gestures)*len(orients),len(gestures))]  # data groupings -> eg. group together different orientations to see performance of gesture classification
# data_groupings = [list(range(x,x+len(gestures))) for x in range(offset,offset+len(gestures)*len(orients),len(gestures))]
# this one is for gesures classes # data_groupings = [list(range(x,offset+len(gestures)*len(orients),len(gestures))) for x in range(offset,offset+len(gestures))]  # data groupings -> eg. group together different orientations to see performance of gesture classification
data_groupings = [list(range(x,offset+len(gestures)*len(orients),len(gestures))) for x in range(offset,offset+len(gestures))]  # data groupings -> eg. group together different orientations to see performance of gesture classification
# this one is for combined orientation and gesture classes # data_groupings = [[x] for x in range(1,17)]
# data_groupings = [[x] for x in range(1,17)]
if include_nomove:
    data_groupings = [[0]] + data_groupings # include [0] for the nomove class

# get data outside of loop
csi, labels = get_csi(dir, subjects, APs, orients, gestures, include_nomove=include_nomove)
x, y = combine_ap_data(csi, labels, len(APs))
y = simplify_labels(y)
y = group_data(y, subject_groupings)
y = group_data(y, data_groupings)
y = simplify_labels(y)
x_data, y_data = normalize_data(x, y)

# print("loading")
# x_data = np.load('file.npy') # load
# y_data = np.load('file.npy') # load
# print("loaded")

# list of alpha values
alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] # centered around 4x

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
    test_MSE = []
    val_B_ACC = []

    # make directory
    if not os.path.exists(f"./alpha_results/{str(alpha)}"):
        os.makedirs(f"./alpha_results/{str(alpha)}")

    print(f"Starting Model Training {a_idx+1}/{len(alphas)}")
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=runs).start()
    for run in range(runs):
        # shuffle and split data each iteration
        x_temp, y_temp, x_test, y_test = shuffle_split_data(x_data, y_data, split=0.8)
        x_train, y_train, x_val, y_val = shuffle_split_data(x_temp, y_temp, split=0.75)
        len_train = x_train.shape[0]
        len_test = x_test.shape[0]

        # expand dims to add single channel
        x_train = np.expand_dims(x_train, axis=3)
        x_val = np.expand_dims(x_val, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

        # caphar
        input = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=16, strides=4, activation="relu",
                                padding='valid', kernel_regularizer='l2', bias_regularizer='l2')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=2, activation='relu',
                                padding='valid', kernel_regularizer='l2', bias_regularizer='l2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        pc = PrimaryCaps(F=16, K=8, N=2808, D=8, s=2)(x)
        x = tf.keras.layers.Dropout(rate=0.5)(pc)
        activity_caps = FCCaps(len(data_groupings),16)(x)
        output = Length(name='length_capsnet_output')(activity_caps)

        capsnet = Model(inputs=[input], outputs=[pc, activity_caps, output])
        
        # generator graph using sub pixel convolution
        input_gen = tf.keras.Input(16*len(data_groupings))
        x_gen = tf.keras.layers.Dense(units=25*48*1, activation="relu")(input_gen)
        x_gen = tf.keras.layers.Reshape(target_shape=(25, 48, 1))(x_gen)
        R = 20
        x_gen = tf.keras.layers.Conv2D(filters=R**2, kernel_size=5, padding='same')(x_gen)
        x_gen = tf.nn.depth_to_space(x_gen, R) # 500x960x1 output
        generator = Model(inputs=input_gen, outputs=x_gen, name='Generator')

        # define combined models
        inputs = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        y_true = tf.keras.Input(shape=(y_train.shape[1]))
        pc, out_caps, out_caps_len = capsnet(inputs)
        masked_by_y = Mask()([out_caps, y_true])  
        masked = Mask()(out_caps)

        x_gen_train = generator(masked_by_y)
        x_gen_eval = generator(masked)

        model = tf.keras.models.Model([inputs, y_true], [out_caps_len, x_gen_train], name='CapsNet_Generator')
        model_test = tf.keras.models.Model(inputs, [out_caps_len, x_gen_eval], name='CapsNet_Generator')

        if run == 0 and a_idx == 0:
            capsnet.summary()
            generator.summary()
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
        history = model.fit([x_train, y_train], [y_train, x_train], batch_size=batch_size, epochs=epochs,
                validation_data=[[x_val, y_val], [y_val, x_val]], callbacks=callbacks, shuffle=True, verbose=verbose)
        training_times.append((time.time() - t_start))
        hists.append(history)

        try:
            val_B_ACC.append(history.history[list(history.history.keys())[-3]][-1])
        except:
            print("val_BAcc Didn't work...")

        model.save(f"./alpha_results/{str(alpha)}/final_model.hdf5")
        # model_test.save('./results/final_model_test.hdf5')
        capsnet.save(f"./alpha_results/{str(alpha)}/final_model_capsnet.hdf5")
        # generator.save('./results/final_model_generator.hdf5')
        
        # if run == 0:
        #     plot_model(history)

        pred, reconst, inference_time = load_model_predictions_generator(x_test, batch_size, model_test)
        inference_times.append(inference_time)

        # get performance metrics
        matrices.append(create_conf_matrix(y_test, pred))
        MCC.append(matthews_corrcoef(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        F1.append(f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), average='micro'))
        B_ACC.append(balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        ACC.append(accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        test_MSE.append(mean_squared_error(x_test.reshape(x_test.shape[0],-1), reconst.reshape(x_test.shape[0],-1)))
        
        print('Test acc:', [round(b,3) for b in B_ACC])
        print('Val acc:', [round(b,3) for b in val_B_ACC])

        del x_train, y_train, x_test, y_test, x_val, y_val, x_temp, y_temp, \
            input, x, pc, activity_caps, output, input_gen, x_gen, inputs, y_true, out_caps, out_caps_len, \
            masked_by_y, masked, capsnet, generator, model, x_gen_train, x_gen_eval, model_test, \
            adam, log, lr_decay_cb, callbacks
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
    f_res.write(str(np.mean(training_times) / epochs / len_train))
    f_res.write("\nInference times:")
    f_res.write(str(inference_times))
    f_res.write("\nAvg inference time:")
    f_res.write(str(np.mean(inference_times)))
    f_res.write("\nAvg inference time per sample:")
    f_res.write(str(np.mean(inference_times) / len_test))
    f_res.write("\nReconst MSE:")
    f_res.write(str(test_MSE))
    f_res.write("\nAvg Reconst MSE:")
    f_res.write(str(np.mean(test_MSE)))
    f_res.write("\n")
    test_MSE
    for hist in hists:
        f_res.write("Single run results:\n")
        for value in hist.history:
            f_res.write(str(value))
            f_res.write(str(hist.history[value]))
            f_res.write("\n")
    f_res.close()

    pbar.finish()

    if len(data_groupings) == 5:
        plot_conf_matrix_5_file(avg_matrix, f"./alpha_results/{str(alpha)}/conf_matrix.png")
    elif len(data_groupings) == 4:
        plot_conf_matrix_4_file(avg_matrix, f"./alpha_results/{str(alpha)}/conf_matrix.png")
    else:
        print(avg_matrix)