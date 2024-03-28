import numpy as np
import os
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from librosa import display
from scipy.io import wavfile
from scipy import fft, signal
from tensorflow.keras import backend as K


class PSD_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512], name="PSD", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        loss = 0
        for i in range(len(self.m)):
            pad_amount = int(self.m[i] // 2)
            #pads = [[0, 0] for _ in range(len(y_true.shape))]
            #pads[0] = [pad_amount, pad_amount]
            pads = [[pad_amount, pad_amount]]
            y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
            y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

            Y_true = K.abs(tf.signal.rfft(y_true))
            Y_pred = K.abs(tf.signal.rfft(y_pred))
            Y_true = K.pow(Y_true, 2)
            Y_pred = K.pow(Y_pred, 2)

            #l_true = K.log(Y_true + 0.0001)
            #l_pred = K.log(Y_pred + 0.0001)

            loss += tf.norm((Y_true - Y_pred), ord=1)/Y_pred.shape[0]
            #loss += tf.norm((l_true - l_pred), ord=1)/Y_pred.shape[0]
        return loss/len(self.m)

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}



class MyLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, training_steps):
        self.initial_learning_rate = initial_learning_rate
        self.steps = training_steps
    def __call__(self, step):
        lr = tf.cast(self.initial_learning_rate * (0.25 ** (tf.cast(step/self.steps, dtype=tf.float32))), dtype=tf.float32)
        return tf.math.minimum(lr, 1e-7)


def writeResults(results, b_size, learning_rate, model_save_dir, save_folder,
                 index):
    results = {
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss']
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(preds, tars, model_save_dir, save_folder, title):

    fs = 24000
    N_fft = fs*2
    for i in range(tars.shape[0]):
        
        tar = tars[i].reshape(-1)
        pred = preds[i].reshape(-1)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        display.waveshow(tar, sr=fs, ax=ax, label='Target', alpha=0.9)
        display.waveshow(pred, sr=fs, ax=ax, label='Prediction', alpha=0.7)
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' +save_folder + '/' + title + str(i) +'plot.pdf', format='pdf')
        plt.close('all')


        #FFT
        FFT_t = np.abs(fft.fftshift(fft.fft(tar, n=N_fft))[N_fft // 2:])
        FFT_p = np.abs(fft.fftshift(fft.fft(pred, n=N_fft))[N_fft // 2:])
        freqs = fft.fftshift(fft.fftfreq(N_fft) * fs)
        freqs = freqs[N_fft // 2:]

        fig, ax = plt.subplots(1, 1)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_t)), label='Target',)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_p)), label='Prediction')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.axis(xmin=20, xmax=22050)

        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/' + title + str(i) + 'FFT.pdf', format='pdf')
        plt.close('all')
        
        tar = np.pad(tar, [0, 1200-len(tar)])
        pred = np.pad(pred, [0, 1200-len(pred)])

        idct_t = scipy.fftpack.idct(tar, type=2, norm='ortho')
        idct_p = scipy.fftpack.idct(pred, type=2, norm='ortho')
        idct_t = scipy.signal.resample_poly(idct_t, 4, 1)
        idct_p = scipy.signal.resample_poly(idct_p, 4, 1)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        display.waveshow(idct_t, sr=fs, ax=ax, label='Target', alpha=0.9)
        display.waveshow(idct_p, sr=fs, ax=ax, label='Prediction', alpha=0.7)
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/' + title + str(i) + 'IDCT.pdf', format='pdf')
        plt.close('all')

def plotTraining(loss_training, loss_val, model_save_dir, save_folder):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/loss.png')
    plt.close('all')


def predictWaves(predictions, y_tests, model_save_dir, save_folder, fs, title):
    
    for i in range(y_tests.shape[0]):
        
        tar = y_tests[i]
        pred = predictions[i] 
        
        pred_name = str(i) + 'notransform_pred.wav'
        tar_name = str(i) + 'notransform_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))
        pred = pred.reshape(-1)
        tar = tar.reshape(-1)
        wavfile.write(pred_dir, fs, pred)
        wavfile.write(tar_dir, fs, tar)

        tar = np.pad(tar, [0, 1200-len(tar)])
        pred = np.pad(pred, [0, 1200-len(tar)])

        idct_t = scipy.fftpack.idct(tar, type=2, norm='ortho')
        idct_p = scipy.fftpack.idct(pred, type=2, norm='ortho')
        idct_t = scipy.signal.resample_poly(idct_t, 4, 1)
        idct_p = scipy.signal.resample_poly(idct_p, 4, 1)
        
        pred_name = str(i) + '_idct_pred.wav'
        tar_name = str(i) +'_idct_tar.wav'
        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))
        idct_p = idct_p.reshape(-1)
        idct_t = idct_t.reshape(-1)
        wavfile.write(pred_dir, fs, idct_p)
        wavfile.write(tar_dir, fs, idct_t)



def plotResult_(pred, tar, model_save_dir, save_folder, title):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(tar, label='tar')
    ax.plot(pred, label='pred')
    ax.legend(loc='upper right')
    fig.savefig(model_save_dir + '/' + save_folder + '/' + title + 'plot.pdf', format='pdf')
    plt.close('all')

def checkpoints(model_save_dir, save_folder, name):
    ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(
        os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
        os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1)
    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=False, save_weights_only=True,
                                                              verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest