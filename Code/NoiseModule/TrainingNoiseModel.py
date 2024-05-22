import os
import tensorflow as tf
from LossFunctions import STFT_loss
from DatasetsClass import DataGeneratorPickles
from NoiseModel import NoiseModel
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler, render_results
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

def train(data_dir, **kwargs):
    learning_rate = kwargs.get('learning_rate', 3e-4)
    num_steps = kwargs.get('num_steps', 240)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    batch_size = kwargs.get('batch_size', 1)

    #tf.keras.backend.set_floatx('float64')
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)
   
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    training_steps = 425600
    max_S = 140000#162000
    w = [1., 1., 1.]
    epochs = 50
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)

    model = NoiseModel(B=batch_size, num_steps=num_steps, max_steps=(max_S / num_steps) - 1, type=tf.float32)

    train_gen = DataGeneratorPickles(data_dir, set='train', steps=num_steps, model=model, batch_size=batch_size)
    test_gen = DataGeneratorPickles(data_dir, set='val', steps=num_steps, model=model, batch_size=batch_size)

    lossesName = ['output_1', 'output_2', 'output_3']
    losses = {
        lossesName[0]: "mse", # rms
        lossesName[1]: STFT_loss(m=[32, 64, 128, 256, 512]),# noise
        lossesName[2]: "mse", # mean
    }
    lossWeights = {lossesName[0]: w[0], lossesName[1]: w[1], lossesName[1]: w[2]}
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt)


    print('learning_rate:', learning_rate)
    print('\n')
    print('num_steps:', num_steps)
    print('\n')
    print('batch_size:', batch_size)
    print('\n')

    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest]
    if not inference:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best).expect_partial()
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        count = 0
        for i in range(epochs):
            print('epoch: ', i)
            model.reset_states()
            results = model.fit(train_gen,
                                shuffle=False,
                                validation_data=test_gen,
                                epochs=1,
                                verbose=0,
                                callbacks=callbacks)
            print(model.optimizer.learning_rate)
            loss_training[i] = (results.history['loss'])[-1]
            loss_val[i] = (results.history['val_loss'])[-1]
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
            else:
                count = count + 1
                if count == 10:
                    break
            if i % 10 == 0:
                model.reset_states()
                pred = model.predict(test_gen, verbose=0)
                render_results(pred[1], test_gen.rms, test_gen.N, model_save_dir, save_folder)

        writeResults(results, batch_size, learning_rate, model_save_dir, save_folder, 1)
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()

    test_loss = model.evaluate(test_gen,
                               verbose=0,
                               return_dict=True)
    results = {'test_loss': test_loss}
    model.reset_states()
    pred = model.predict(test_gen, verbose=0)
    model.reset_states()
    render_results(pred[1], test_gen.rms, test_gen.N, model_save_dir, save_folder)

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return 42
