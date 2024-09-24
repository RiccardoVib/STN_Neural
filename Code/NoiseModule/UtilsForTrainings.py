import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from librosa import display
from scipy.io import wavfile
import librosa.display
from scipy import fft, signal
from Utils import filterAudio

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


def writeResults(results, units, epochs, b_size, learning_rate, model_save_dir,
                 save_folder,
                 index):
    """
    write to a text the result and parameters of the training
      :param results: the results from the fit function [dictionary]
      :param units: the number of model's units [int]
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
        'units': units,
        'epochs': epochs
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(pred, tar, model_save_dir, save_folder, title):
    """
    Plot the rendered results
      :param pred: the model's prediction  [array of floats]
      :param tar: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param title: the name of the file [string]
      """
    fs = 24000
    N_stft = 2048
    N_fft = fs * 2

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(tar, label='Target', alpha=0.9)
    ax.plot(pred, label='Prediction', alpha=0.7)
    # ax.label_outer()
    ax.legend(loc='upper right')
    fig.savefig(model_save_dir + '/' + save_folder + '/' + title + 'plot')
    plt.close('all')

    # FFT
    FFT_t = np.abs(fft.fftshift(fft.fft(tar, n=N_fft))[N_fft // 2:])
    FFT_p = np.abs(fft.fftshift(fft.fft(pred, n=N_fft))[N_fft // 2:])
    freqs = fft.fftshift(fft.fftfreq(N_fft) * fs)
    freqs = freqs[N_fft // 2:]

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_t)), label='Target', )
    ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_p)), label='Prediction')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.axis(xmin=20, xmax=22050)

    ax.legend(loc='upper right')
    fig.savefig(model_save_dir + '/' + save_folder + '/' + title + 'FFT.pdf', format='pdf')
    plt.close('all')

    # STFT
    D = librosa.stft(tar, n_fft=N_stft)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax[0].pcolormesh(t, f, np.abs(Zxx), vmin=np.min(np.abs(Zxx)), vmax=np.max(np.abs(Zxx)), shading='gouraud')
    librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
    ax[0].set_title('STFT Magnitude (Top: target, Bottom: prediction)')
    ax[0].set_ylabel('Frequency [Hz]')
    ax[0].set_xlabel('Time [sec]')
    ax[0].label_outer()

    # f, t, Zxx = signal.stft(_p, fs, nperseg=1000)
    D = librosa.stft(pred, n_fft=N_stft)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[1])
    # ax[1].pcolormesh(t, f, np.abs(Zxx), vmin=np.min(np.abs(Zxx)), vmax=np.max(np.abs(Zxx)), shading='gouraud')
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    fig.savefig(model_save_dir + '/' + save_folder + '/' + title + 'STFT.pdf', format='pdf')
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


def predictWaves(predictions, y_test, model_save_dir, save_folder, fs, title):
    """
    Render the prediction, target as wav audio file
      :param predictions: the model's prediction  [array of floats]
      :param y_test: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param fs: the sampling rate [int]
      :param title: the name of the file [string]
    """
    pred_name = title + '_pred.wav'
    tar_name = title + '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    wavfile.write(pred_dir, fs, predictions.reshape(-1))
    wavfile.write(tar_dir, fs, y_test.reshape(-1))

    plotResult(predictions.reshape(-1), y_test.reshape(-1), model_save_dir, save_folder, title)



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
    ax.plot(pred, label='pred')
    ax.plot(tar, label='tar')
    ax.legend(loc='upper right')
    fig.savefig(model_save_dir + '/' + save_folder + '/' + title + 'plot')
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



def render_results(preds, N, model_save_dir, save_folder):
    """
    Render the prediction, target as wav audio file
      :param preds: the model's prediction  [array of floats]
      :param N: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
    """
    predictions = np.array(preds.reshape(-1), dtype=np.float32)
    N = np.array(N, dtype=np.float32)
    predictWaves(predictions, N, model_save_dir, save_folder, 24000, 'N')
