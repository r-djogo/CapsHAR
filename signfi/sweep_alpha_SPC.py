import os
import gc
import time
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import shuffle

from HAR_CSI import *
from utils.layers import FCCaps, Length, PrimaryCaps, Mask
from utils.tools import marginLoss

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Parameters
lr, lr_decay, batch_size, epochs = 5e-4, 0.97, 128, 50  # model training hyperparameters
verbose = 0 # 0 = silent, 1 = progress bar, 2 = one line per epoch
n_steps = 200
n_input = 90
n_classes = 276 # WiFi activity total classes
only_lab, only_home, both = False, False, True
if only_lab ^ only_home ^ both and not (only_home and only_lab and both):
    pass
else:
    raise Exception('Choose only 1 of "only_lab", "only_home", and "both"')

print("loading data...")
x_data_home = np.load('x_data_home.npy') # load
y_data_home = np.load('y_data_home.npy') # load
x_data_lab = np.load('x_data_lab.npy') # load
y_data_lab = np.load('y_data_lab.npy') # load
print("data loaded")

# list of alpha values
alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45] # centered around 4x

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

    # make directory
    if not os.path.exists(f"./alpha_results/{str(alpha)}"):
        os.makedirs(f"./alpha_results/{str(alpha)}")

    print(f"Starting Model Training {a_idx+1}/{len(alphas)}")
    
    # create dataset
    if only_lab:
        x_data = x_data_lab
        y_data = y_data_lab
    elif only_home:
        x_data = x_data_home
        y_data = y_data_home
    elif both:
        x_data = np.concatenate((x_data_lab, x_data_home), axis=0)
        y_data = np.concatenate((y_data_lab, y_data_home), axis=0)
    # data shuffle
    x_data, y_data = shuffle(x_data, y_data, random_state=0)

    # k_fold
    kk = 5
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=kk).start()

    for i in range(kk):
        #Roll the data
        x_data = np.roll(x_data, int(len(x_data) / kk), axis=0)
        y_data = np.roll(y_data, int(len(y_data) / kk), axis=0)

        #data separation
        wifi_x_train = x_data[int(len(x_data) / kk):]
        wifi_y_train = y_data[int(len(y_data) / kk):]
        # one-hot
        wifi_y_train = np.eye(n_classes)[np.squeeze(wifi_y_train)-1]

        wifi_x_validation = x_data[:int(len(x_data) / kk)]
        wifi_y_validation = y_data[:int(len(y_data) / kk)]
        # one-hot
        wifi_y_validation = np.eye(n_classes)[np.squeeze(wifi_y_validation)-1]

        # expand dims to add single channel
        wifi_x_train = np.expand_dims(wifi_x_train, axis=3)
        wifi_x_validation = np.expand_dims(wifi_x_validation, axis=3)
        # print(wifi_x_train.shape, wifi_y_train.shape, wifi_x_validation.shape, wifi_y_validation.shape)
        len_train = wifi_x_train.shape[0]
        len_test = wifi_x_validation.shape[0]

        wifi_x_train = wifi_x_train.astype(np.float32)
        wifi_y_train = wifi_y_train.astype(np.float32)
        wifi_x_validation = wifi_x_train.astype(np.float32)
        wifi_y_validation = wifi_y_train.astype(np.float32)

        # capshar
        input = tf.keras.Input(shape=(n_steps, n_input, 1), dtype='float32') #(200,90,1)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=16, strides=2, activation="relu",
                                padding='valid', kernel_regularizer='l2', bias_regularizer='l2')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=2, activation='relu',
                                padding='valid', kernel_regularizer='l2', bias_regularizer='l2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = PrimaryCaps(F=16, K=8, N=180, D=8, s=2)(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        dim_activity_caps = 8
        activity_caps = FCCaps(n_classes,dim_activity_caps)(x)
        output = Length(name='length_capsnet_output')(activity_caps)

        capsnet = Model(inputs=[input], outputs=[activity_caps, output])

        # generator graph using sub pixel convolution
        input_gen = tf.keras.Input(dim_activity_caps*n_classes, dtype='float32')
        x_gen = tf.keras.layers.Dense(units=20*9*1, activation="relu")(input_gen)
        x_gen = tf.keras.layers.Reshape(target_shape=(20, 9, 1))(x_gen)
        R = 10
        x_gen = tf.keras.layers.Conv2D(filters=R**2, kernel_size=5, padding='same')(x_gen)
        x_gen = tf.nn.depth_to_space(x_gen, R) #(200,90,1) output
        generator = Model(inputs=input_gen, outputs=x_gen, name='Generator')

        # define combined models
        inputs = tf.keras.Input(shape=(n_steps, n_input, 1), dtype='float32')
        y_true = tf.keras.Input(shape=(wifi_y_train.shape[1]))
        out_caps, out_caps_len = capsnet(inputs)
        masked_by_y = Mask()([out_caps, y_true])  
        masked = Mask()(out_caps)

        x_gen_train = generator(masked_by_y)
        x_gen_eval = generator(masked)

        model = tf.keras.models.Model([inputs, y_true], [out_caps_len, x_gen_train], name='CapsNet_Generator')
        model_test = tf.keras.models.Model(inputs, [out_caps_len, x_gen_eval], name='CapsNet_Generator')

        if i == 0 and a_idx == 0:
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
        history = model.fit([wifi_x_train, wifi_y_train], [wifi_y_train, wifi_x_train], batch_size=batch_size, epochs=epochs,
                validation_data=[[wifi_x_validation, wifi_y_validation], [wifi_y_validation, wifi_x_validation]],
                callbacks=callbacks, shuffle=True, verbose=verbose)
        training_times.append((time.time() - t_start))
        hists.append(history)

        model.save(f"./alpha_results/{str(alpha)}/final_model.hdf5")
        # capsnet.save(f"./alpha_results/{str(alpha)}/final_model_capsnet.hdf5')

        # if run == 0:
        #     plot_model(history)

        pred, inference_time = load_model_predictions_generator(wifi_x_validation, batch_size, model_test)
        inference_times.append(inference_time)

        # get performance metrics
        matrices.append(create_conf_matrix(wifi_y_validation, pred))
        MCC.append(matthews_corrcoef(np.argmax(wifi_y_validation, axis=1), np.argmax(pred, axis=1)))
        F1.append(f1_score(np.argmax(wifi_y_validation, axis=1), np.argmax(pred, axis=1), average='micro'))
        B_ACC.append(balanced_accuracy_score(np.argmax(wifi_y_validation, axis=1), np.argmax(pred, axis=1)))
        ACC.append(accuracy_score(np.argmax(wifi_y_validation, axis=1), np.argmax(pred, axis=1)))

        print('Test acc:', [round(b,3) for b in B_ACC])

        del wifi_x_train, wifi_y_train, wifi_x_validation, wifi_y_validation, \
            input, x, activity_caps, output, input_gen, x_gen, inputs, y_true, out_caps, out_caps_len, \
            masked_by_y, masked, capsnet, generator, model, x_gen_train, x_gen_eval, model_test, \
            adam, log, lr_decay_cb, callbacks
        gc.collect()

        pbar.update(i+1)

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
    f_res.write("\n")
    for hist in hists:
        f_res.write("Single run results:\n")
        for value in hist.history:
            f_res.write(str(value))
            f_res.write(str(hist.history[value]))
            f_res.write("\n")
    f_res.close()

    print(avg_matrix)

    pbar.finish()