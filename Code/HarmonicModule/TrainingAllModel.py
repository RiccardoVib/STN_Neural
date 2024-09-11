import os
import tensorflow as tf
from NFLossFunctions import centLoss, STFT_loss, NMSELoss
from DatasetsClass import DataGeneratorPickles
from PianoModel import PianoModel
from UtilsForTrainings import plotTraining, writeResults, checkpoints, render_results, render_results_train, MyLRScheduler
import pickle
import random
import numpy as np

def train(data_dir, **kwargs):
    learning_rate = kwargs.get('learning_rate', 3e-4)
    harmonics = kwargs.get('harmonics', 24)
    num_steps = kwargs.get('num_steps', 240)
    model_save_dir = kwargs.get('model_save_dir', '../../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    phase = kwargs.get('phase', 'B')
    batch_size = kwargs.get('batch_size', None)
    minibatch_size = kwargs.get('minibatch_size', None)
    inference = kwargs.get('inference', False)
    filename = kwargs.get('filename', '')
    g = kwargs.get('g', 9)
    phan = kwargs.get('phan', False)
    epochs = kwargs.get('epochs', 10)

    # tf.keras.backend.set_floatx('float64')
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)
    
    fs = 24000
    num_samples = batch_size*minibatch_size*num_steps
    fft_size = int(2 ** np.ceil(np.log2(num_samples)))

    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    training_steps = 300000*5

    if phase == 'B':
        w = [1.]
        train_B = True
        train_amps = False
        train_S = False

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

        lossesName = ['output_1']
        losses = {
            lossesName[0]: centLoss(delta=1),  # partials
        }

        lossWeights = {lossesName[0]: w[0]
                    }

    elif phase == 'A':
        w = [0.5, 0., 1.]
        train_B = False
        train_amps = True
        train_S = False
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps))

        lossesName = ['output_1', 'output_2', 'output_3']
        losses = {
            lossesName[0]: STFT_loss(m=[256, 512, 1024, 2048, 4096, 8192], fft_size=fft_size, num_samples=num_samples), # S
            lossesName[1]: 'mae',  # rms
            lossesName[2]: 'mae',  # alfa
        }

        lossWeights = {lossesName[0]: w[0],
                       lossesName[1]: w[1],
                       lossesName[2]: w[2],
                       }


    test_gen = DataGeneratorPickles(filename, data_dir, set='val', steps=num_steps, model=None,
                                    batch_size=batch_size,
                                    minibatch_size=minibatch_size, stage=phase)

    model = PianoModel(batch_size=batch_size,
                       num_steps=num_steps,
                       fs=fs,
                       g=g,
                       phan=phan,
                       num_frames=minibatch_size,
                       harmonics=harmonics,
                       max_steps=(test_gen.ratio - 1),
                       train_b=train_B,
                       train_amps=train_amps,
                       train_S=train_S,
                       type=tf.float32)


    model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt)
    
    print('learning_rate:', learning_rate)
    print('\n')
    print('harmonics:', harmonics)
    print('\n')
    print('phase:', phase)
    print('\n')
    print('batch_size:', batch_size)
    print('\n')
    print('minibatch_size:', minibatch_size)
    print('\n')
    print('training_steps:', training_steps)
    print('\n')
    print('g:', g)
    print('\n')
    print('phan:', phan)
    print('\n')
    
    train_gen = DataGeneratorPickles(filename, data_dir, set='train', steps=num_steps, model=model,
                                     batch_size=batch_size,
                                     minibatch_size=minibatch_size, stage=phase)
    test_gen = DataGeneratorPickles(filename, data_dir, set='val', steps=num_steps, model=model,
                                    batch_size=batch_size,
                                    minibatch_size=minibatch_size, stage=phase)

    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest]
    if not inference:
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last).expect_partial()
            #start_epoch = int(latest.split('-')[-1].split('.')[0])
            #print('Starting from epoch: ', start_epoch + 1)
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
            
            #print('b: ', model.DecayModel.b)
            #print('alfas: ', model.DecayModel.alfas)

            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                count = count + 1
                if count == 50:
                    break

        writeResults(results, batch_size, learning_rate, model_save_dir, save_folder, 1)
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

    if phase == 'B':
        #render_results(pred, test_gen.S, model_save_dir, save_folder)

        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'B_results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)

        #model.reset_states()
        #pred = model.predict(train_gen, verbose=0)
        #render_results_train(pred, train_gen.S, model_save_dir, save_folder)
  
    else:
        render_results(pred[0], test_gen.S, model_save_dir, save_folder, phan)

        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'A_results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)

        model.reset_states()
        pred = model.predict(train_gen, verbose=0)
        render_results_train(pred[0], train_gen.S, model_save_dir, save_folder, phan)
    return 42

