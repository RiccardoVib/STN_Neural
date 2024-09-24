import numpy as np
import os
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
from librosa import display
from scipy.io import wavfile
from scipy import fft, signal
from tensorflow.keras import backend as K



class STFT_loss(tf.keras.losses.Loss):
    """ multi-STFT error """

    def __init__(self, m=[32, 64, 128, 256, 512, 1024], name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])

        loss = 0

        for i in range(len(self.m)):
            pad_amount = int(self.m[i] // 2)  # Symmetric even padding like librosa.
            pads = [[0, 0] for _ in range(len(y_true.shape))]
            pads[1] = [pad_amount, pad_amount]
            y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
            y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

            Y_true = K.abs(
                tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))
            Y_pred = K.abs(
                tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))

            Y_true = K.pow(Y_true, 2)
            Y_pred = K.pow(Y_pred, 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.norm((l_true - l_pred), ord=1) + tf.norm((Y_true - Y_pred), ord=1)

        return loss / len(self.m)

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}





class MyLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Define the learning schedule
      :param initial_learning_rate: the initial learning rate [float]
      :param training_steps: the number of total training steps (iterations) [int]
    """
    def __init__(self, initial_learning_rate, training_steps):
        self.initial_learning_rate = initial_learning_rate
        self.steps = training_steps * 30

    def __call__(self, step):
        lr = tf.cast(self.initial_learning_rate * (0.25 ** (tf.cast(step / self.steps, dtype=tf.float32))),
                     dtype=tf.float32)
        return lr#tf.math.maximum(lr, 1e-6)


def writeResults(results, epochs, b_size, learning_rate, model_save_dir,
                 save_folder,
                 index):
    """
    write to a text the result and parameters of the training
      :param results: the results from the fit function [dictionary]
      :param epochs: the number of epochs [int]
      :param b_size: the batch size [int]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param index: index for naming the file [string]

    """
    results = {
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss'],
        'epochs': epochs
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)


def plotResult(preds, tars, ref, model_save_dir, save_folder, title):
    """
    Plot the rendered results
      :param pred: the model's prediction  [array of floats]
      :param tar: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param title: the name of the file [string]
      """

    fs = 24000
    N_fft = fs * 2
    for i in range(tars.shape[0]):
        tar = tars[i].reshape(-1)
        pred = preds[i].reshape(-1)
        ref_v = ref[i].reshape(-1)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        display.waveshow(tar, sr=fs, ax=ax, label='Target', alpha=0.9)
        # display.waveshow(ref, sr=fs, ax=ax, label='ref', alpha=0.9)
        display.waveshow(pred, sr=fs, ax=ax, label='Prediction', alpha=0.7)
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/' + title + str(i) + 'plot.pdf', format='pdf')
        plt.close('all')

        # FFT
        FFT_t = np.abs(fft.fftshift(fft.fft(tar, n=N_fft))[N_fft // 2:])
        FFT_p = np.abs(fft.fftshift(fft.fft(pred, n=N_fft))[N_fft // 2:])
        # FFT_r = np.abs(fft.fftshift(fft.fft(ref_v, n=N_fft))[N_fft // 2:])
        freqs = fft.fftshift(fft.fftfreq(N_fft) * fs)
        freqs = freqs[N_fft // 2:]

        fig, ax = plt.subplots(1, 1)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_t)), label='Target', )
        # ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_r)), label='Target',)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_p)), label='Prediction')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.axis(xmin=20, xmax=22050)

        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/' + title + str(i) + 'FFT.pdf', format='pdf')
        plt.close('all')

        tar = np.pad(tar, [0, 1300 - len(tar)])
        pred = np.pad(pred, [0, 1300 - len(pred)])

        idct_t = scipy.fftpack.idct(tar, type=2, norm='ortho')
        idct_p = scipy.fftpack.idct(pred, type=2, norm='ortho')
        #idct_t = scipy.signal.resample_poly(idct_t, 4, 1)
        #idct_p = scipy.signal.resample_poly(idct_p, 4, 1)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        display.waveshow(idct_t, sr=fs, ax=ax, label='Target', alpha=0.9)
        display.waveshow(ref_v, sr=fs, ax=ax, label='ref', alpha=0.9)
        display.waveshow(idct_p, sr=fs, ax=ax, label='Prediction', alpha=0.7)
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/' + title + str(i) + 'IDCT.pdf', format='pdf')
        plt.close('all')


def plotTraining(loss_training, loss_val, model_save_dir, save_folder, name):
    """
    Plot the training against the validation losses
      :param loss_training: vector with training losses [array of floats]
      :param loss_val: vector with validation losses [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param fs: the sampling rate [int]
      :param filename: the name of the file [string]
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')
    plt.legend(loc='upper center')  # , bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/' + name + 'loss.png')
    plt.close('all')

def predictWaves(predictions, y_tests, ref_v, model_save_dir, save_folder, fs):
    """
    Render the prediction, target as wav audio file
      :param predictions: the model's prediction  [array of floats]
      :param y_test: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param fs: the sampling rate [int]
      :param title: the name of the file [string]
    """

    for i in range(y_tests.shape[0]):

        tar = y_tests[i]
        pred = predictions[i]
        ref = ref_v[i]

        pred_name = str(i) + 'notransform_pred.wav'
        tar_name = str(i) + 'notransform_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))
        pred = np.array(pred, dtype=np.float32).reshape(-1)
        tar = np.array(tar, dtype=np.float32).reshape(-1)
        wavfile.write(pred_dir, fs, pred)
        wavfile.write(tar_dir, fs, tar)

        tar = np.pad(tar, [0, 1300 - len(tar)])
        pred = np.pad(pred, [0, 1300 - len(pred)])
        idct_t = scipy.fftpack.idct(tar, type=2, norm='ortho')
        idct_p = scipy.fftpack.idct(pred, type=2, norm='ortho')
        #idct_t = scipy.signal.resample_poly(idct_t, 4, 1)
        #idct_p = scipy.signal.resample_poly(idct_p, 4, 1)

        pred_name = str(i) + '_idct_pred.wav'
        tar_name = str(i) + '_idct_tar.wav'
        ref_name = str(i) + '_idct_ref.wav'
        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))
        ref_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', ref_name))
        idct_p = np.array(idct_p, dtype=np.float32).reshape(-1)
        idct_t = np.array(idct_t, dtype=np.float32).reshape(-1)
        ref = np.array(ref, dtype=np.float32).reshape(-1)
        wavfile.write(pred_dir, fs, idct_p)
        wavfile.write(tar_dir, fs, idct_t)
        wavfile.write(ref_dir, fs, ref)


def plotResult_(pred, tar, model_save_dir, save_folder, title):
    """
    plot the prediction and target
      :param pred: the model's prediction  [array of floats]
      :param tar: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param title: the name of the file [string]
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(tar, label='tar')
    ax.plot(pred, label='pred')
    ax.legend(loc='upper right')
    fig.savefig(model_save_dir + '/' + save_folder + '/' + title + 'plot.pdf', format='pdf')
    plt.close('all')


def checkpoints(model_save_dir, save_folder):
    """
    Define the path to the checkpoints saving the last and best epoch's weights
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
    """
    ckpt_path = os.path.normpath(
        os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
       os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1,
                                                       save_best_value=True)

    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss', mode='min', save_best_only=False, save_weights_only=True, verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest