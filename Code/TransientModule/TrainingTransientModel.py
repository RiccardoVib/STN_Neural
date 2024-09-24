import os
import tensorflow as tf
from DatasetsClass import DataGeneratorPickles
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, plotResult, STFT_loss
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


def train(data_dir, **kwargs):
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param filename: name of the dataset [bool]
    """

    batch_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 3e-4)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    filename = kwargs.get('filename', '')
    epochs = 1000  # 2000

    # tf.keras.backend.set_floatx('float64')
    # set all the seed in case reproducibility is desired
    # np.random.seed(422)
    # tf.random.set_seed(422)
    # random.seed(422)

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)

    # create the DataGenerator object to retrieve the data
    train_gen = DataGeneratorPickles(filename, data_dir, set='train', batch_size=batch_size)
    test_gen = DataGeneratorPickles(filename, data_dir, set='val',  batch_size=batch_size)


    # define the Adam optimizer with the initial learning rate, training steps
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

    # define the model
    ####################
    inputs = tf.keras.Input(batch_shape=(batch_size, 1), name='vel_inputs')
    act = 'tanh'


    out = tf.keras.layers.Dense(32)(inputs)
    out = tf.expand_dims(out, axis=-1)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(2, 2, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(4, 2, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(8, 2, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(8, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)#####
    out = tf.keras.layers.Conv1D(16, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)

    out = tf.keras.layers.Conv1D(8, 4, activation=act, padding='valid')(out)#16
    out = tf.keras.layers.Conv1D(4, 32, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(2, 128, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(2, 256, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(2, 256, activation=act, padding='valid')(out)


    out = tf.keras.layers.Conv1D(1, 4, activation=act, padding='valid')(out)
    model = tf.keras.Model(inputs, out)
    model.summary()
    ############################################################

    model.compile(loss=STFT_loss(m=[32, 64, 128, 256]), metrics=['mse'],
                  optimizer=opt)

    # define callbacks: where to store the weights
    callbacks = []
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=20)
    stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)
    callbacks += [ckpt_callback, ckpt_callback_latest, scheduler, stop]
    if not inference:
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")


        results = model.fit(train_gen,
                            batch_size=batch_size,
                            shuffle=False,
                            epochs=epochs,
                            verbose=0,
                            validation_data=test_gen,
                            callbacks=callbacks)

        print(model.optimizer.learning_rate)
        loss_training = (results.history['loss'])
        loss_val = (results.history['val_loss'])

        # write and save results
        writeResults(results, batch_size, learning_rate, model_save_dir, save_folder, 1)
        # plot the training and validation loss for all the training
        plotTraining(loss_training, loss_val, model_save_dir, save_folder)


    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")


    test_loss = model.evaluate(test_gen,
                               batch_size=batch_size,
                               verbose=0,
                               return_dict=True)
    results = {'test_loss': test_loss}
    pred = model.predict(test_gen, batch_size=batch_size, verbose=0)

    # plot and render the output audio file, together with the input and target
    plotResult(pred, test_gen.DCT_s, test_gen.DCT, model_save_dir, save_folder, 'dct')
    predictWaves(pred, test_gen.DCT_s, test_gen.DCT, model_save_dir, save_folder, 24000)

    # write and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)

    return 42
