import os
import tensorflow as tf
from LossFunctions import centLoss, STFT_loss
from DatasetsClass import DataGeneratorPickles
from PianoModel import PianoModel
from UtilsForTrainings import plotTraining, writeResults, checkpoints, render_results, MyLRScheduler
import pickle
import random
import numpy as np


def train(data_dir, **kwargs):
    b_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 3e-4)
    harmonics = kwargs.get('harmonics', 24)
    num_steps = kwargs.get('num_steps', 240)
    model_save_dir = kwargs.get('model_save_dir', '../../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    phase = kwargs.get('phase', 'B')
    inference = kwargs.get('inference', False)

    # tf.keras.backend.set_floatx('float64')
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)
    fs = 24000

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
   
    batch_size = (336000//num_steps)//32
    training_steps = 425600
    if phase == 'B':
        w = [1., 1., 0., 0., 0., 0.]
        train_B = True
        train_amps = False
        train_rev=False
        epochs = 100
        learning_rate = 3e-4
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

    elif phase == 'A':
        w = [0., 0., 1., 1., 1., 1.]
        train_B = False
        train_amps = True
        train_rev=False
        epochs = 1000
        learning_rate = 3e-4
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)

    elif phase == 'R':
        w = [0., 0., 1., 1., 1., 1]
        train_B = False
        train_amps = False
        train_rev = True
        epochs = 1000
        learning_rate = 3e-4
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)
    
    
    model = PianoModel(B=batch_size,
                       num_steps=num_steps,
                       harmonics=harmonics,
                       fs=fs,
                       max_steps=2799.0,  # 13999.0,
                       train_b=train_B,
                       train_amps=train_amps,
                       train_rev=train_rev,
                       type=tf.float32)

    train_gen = DataGeneratorPickles(data_dir, set='train', steps=num_steps, model=model, batch_size=batch_size)
    test_gen = DataGeneratorPickles(data_dir, set='val', steps=num_steps, model=model, batch_size=batch_size)
    

    lossesName = ['output_1', 'output_2', 'output_3', 'output_4', 'output_5', 'output_6']
    losses = {
        lossesName[0]: centLoss(delta=1),  # partials
        lossesName[1]: "mse",  # B
        lossesName[2]: STFT_loss(m=[32, 64, 128, 256, 512, 1024]),  # S
        lossesName[3]: "mse",  # rms
        lossesName[4]: "mae",  # alfa
        lossesName[5]: "mae", # attack time
    }
    lossWeights = {lossesName[0]: w[0], lossesName[1]: w[1], lossesName[2]: w[2], lossesName[3]: w[3],
                   lossesName[4]: w[4], lossesName[5]: w[5]}

    model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt)

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

        loss_training = []
        loss_val = []
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
            loss_training.append(results.history['loss'])
            loss_val.append(results.history['val_loss'])
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                count = count + 1
                if count == 10:
                    break
            if i % 10 == 0:
                model.reset_states()
                pred = model.predict(test_gen, verbose=0)
                render_results(pred, test_gen.S, test_gen.rms, None, None, model_save_dir,
                               save_folder)

        writeResults(results, b_size, learning_rate, model_save_dir, save_folder, 1)
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()

    model.reset_states()
    test_loss = model.evaluate(test_gen,
                               verbose=0,
                               return_dict=True)
    results = {'test_loss': test_loss}
    model.reset_states()
    pred = model.predict(test_gen, verbose=0)
    model.reset_states()

    render_results(pred, test_gen.S, test_gen.rms, test_gen.alfas, test_gen.attackTimes, model_save_dir, save_folder)

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return 42
