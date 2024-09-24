import os
import tensorflow as tf
from LossFunctions import STFT_loss, STFT_N_loss
from DatasetsClass import DataGeneratorPickles
from NoiseModel import NoiseModel
from UtilsForTrainings import plotTraining, writeResults, checkpoints, MyLRScheduler, render_results
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

def train(data_dir, **kwargs):
    """
    :param data_dir: the directory in which dataset are stored [string]
    :param batch_size: the size of each batch [int]
    :param learning_rate: the initial leanring rate [float]
    :param num_steps: number of timesteps to generate per iteration [int]
    :param model_save_dir: the directory in which models are stored [string]
    :param save_folder: the directory in which the model will be saved [string]
    :param inference: if True it skip the training and it compute only the inference [bool]
    :param epochs: the number of epochs [int]
    :param filename: name of the datatset [string]
    """
    batch_size = kwargs.get('batch_size', 1)
    learning_rate = kwargs.get('learning_rate', 3e-4)
    num_steps = kwargs.get('num_steps', 240)
    model_save_dir = kwargs.get('model_save_dir', '../../../../TubeTechModels/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    filename = kwargs.get('filename', '')
    epochs = 5000

    #tf.keras.backend.set_floatx('float64')
    #tf.keras.backend.set_floatx('float64')
    # set all the seed in case reproducibility is desired
    #np.random.seed(422)
    #tf.random.set_seed(422)
    #random.seed(422)

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)

    # create the DataGenerator object to retrieve the data
    test_gen = DataGeneratorPickles(filename, data_dir, set='val', steps=num_steps, model=None, batch_size=batch_size)
    training_steps = 425600
    max_S = test_gen.ratio

    # define the Adam optimizer with the initial learning rate, training steps
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)

    # create the model
    model = NoiseModel(batch_size=batch_size, num_steps=num_steps, max_steps=(max_S), type=tf.float32)

    # create the DataGenerator object to retrieve the data
    train_gen = DataGeneratorPickles(filename, data_dir, set='train', steps=num_steps, model=model, batch_size=batch_size)
    test_gen = DataGeneratorPickles(filename, data_dir, set='val', steps=num_steps, model=model, batch_size=batch_size)

    # define the loss and the related weights
    w = [1., 1., 1.]
    lossesName = ['output_1', 'output_2', 'output_3']
    losses = {
        lossesName[0]: "mse", # rms
        lossesName[1]: STFT_loss(m=[32, 64, 128]),# noise
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

    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)

    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")

        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        # counting for early stopping
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
            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            # if not count is increased by one and if equal to 50 the training is stopped
            else:
                count = count + 1
                if count == 50:
                    break

        # write and save results
        writeResults(results, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, 1)

        # plot the training and validation loss for all the training
        loss_training = np.array(loss_training[:i])
        loss_val = np.array(loss_val[:i])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

        print("Training done")


    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    test_loss = model.evaluate(test_gen,
                               verbose=0,
                               return_dict=True)

    # reset the states before predicting
    model.reset_states()
    pred = model.predict(test_gen, verbose=0)

    stft = STFT_N_loss(m=[32, 64, 128])(np.array(test_gen.N, dtype=np.float32), pred[1])

    results = {'test_loss': test_loss, 'stft_norm': stft}

    # plot and render the output audio file, together with the input and target
    render_results(pred[1], test_gen.N, model_save_dir, save_folder)

    # write and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)


    return 42
