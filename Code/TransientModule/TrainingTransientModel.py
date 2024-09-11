import os
import tensorflow as tf
from DatasetsClass import DataGeneratorPickles
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, plotResult, STFT_loss
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


def train(data_dir, **kwargs):
    batch_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 3e-4)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    filename = kwargs.get('filename', '')

    # tf.keras.backend.set_floatx('float64')
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    train_gen = DataGeneratorPickles(filename, data_dir, set='train', batch_size=batch_size)
    test_gen = DataGeneratorPickles(filename, data_dir, set='val',  batch_size=batch_size)


    epochs = 1000  # 2000
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)
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

    callbacks = []
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=20)
    stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest, scheduler, stop]
    if not inference:
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last).expect_partial()
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
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

        writeResults(results, batch_size, learning_rate, model_save_dir, save_folder, 1)
        plotTraining(loss_training, loss_val, model_save_dir, save_folder)

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()


    test_loss = model.evaluate(test_gen,
                               batch_size=batch_size,
                               verbose=0,
                               return_dict=True)
    results = {'test_loss': test_loss}
    pred = model.predict(test_gen, batch_size=batch_size, verbose=0)

    plotResult(pred, test_gen.DCT_s, test_gen.DCT, model_save_dir, save_folder, 'dct')
    predictWaves(pred, test_gen.DCT_s, test_gen.DCT, model_save_dir, save_folder, 24000)

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return 42
