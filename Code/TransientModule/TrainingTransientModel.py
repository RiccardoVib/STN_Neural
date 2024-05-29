import os
import tensorflow as tf
from DatasetsClass import DataGeneratorPickles
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler, plotResult, STFT_loss
import pickle
import random
import numpy as np
from TransientModel import create_tmodel

def train(data_dir, **kwargs):
    b_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 3e-4)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)

    #tf.keras.backend.set_floatx('float64')
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    training_steps = 425600
    
    size = 1200//4
    batch_size = 2800
    epochs = 2000

    train_gen = DataGeneratorPickles(data_dir, set='train', size=size, batch_size=batch_size)
    test_gen = DataGeneratorPickles(data_dir, set='val', size=size, batch_size=batch_size)
        

    #opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)

    ####################
    model = create_tmodel()
    ############################################################

    model.compile(loss=STFT_loss(m=[32, 64, 128, 256, 512]), metrics=['mse'], optimizer=opt)

    callbacks = []
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=20)
    #stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=1e-15, mode='min')

    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest, scheduler]#stop]
    if not inference:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best).expect_partial()
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
        loss_val =(results.history['val_loss'])
                   
        writeResults(results, b_size, learning_rate, model_save_dir, save_folder, 1)
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

    plotResult(pred, test_gen.dct, model_save_dir, save_folder, 'dct')
    predictWaves(pred, test_gen.dct, model_save_dir, save_folder, 24000, 'dct')
    
    
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return 42
