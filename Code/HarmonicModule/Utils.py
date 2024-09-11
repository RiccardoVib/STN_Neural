import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from librosa import display
import librosa.display
from scipy import fft
from scipy.signal import butter, lfilter
import tensorflow as tf

def AttTime(x):
    _startThreshold = 0.
    _stopThreshold = 1

    maxvalue = max(x)

    startAttack = 0.0
    cutoffStartAttack = maxvalue * _startThreshold
    stopAttack = 0.0
    cutoffStopAttack = maxvalue * _stopThreshold

    for i in range(len(x)):
        if (x[i] >= cutoffStartAttack):
            startAttack = i
            break

    for i in range(len(x)):
        if (x[i] >= cutoffStopAttack):
            stopAttack = i
            break

    attackStart = startAttack
    attackStop = stopAttack

    attackTime = attackStop - attackStart

    return np.divide(1., attackTime)



def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)

def loadFilePickle(data_dir, filename):

    file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
    Z = pickle.load(file_data)
    return Z

def plotTime(x, fs):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(predictions, label='pred')
    # ax.plot(x, label='inp')
    # ax.plot(y, label='tar')
    display.waveshow(x, sr=fs, ax=ax)

def plotFreq(x, fs, N):

    FFT = np.abs(fft.fftshift(fft.fft(x, n=N))[N // 2:])/N
    freqs = fft.fftshift(fft.fftfreq(N) * fs)
    freqs = freqs[N // 2:]

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(freqs, 20 * np.log10(np.abs(FFT)+1))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.axis(xmin=20, xmax=22050)

def plotSpectogram(x, fs, N):

    D = librosa.stft(x, n_fft=N)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.label_outer()

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def filterAudio(x, min_f, max_f, fs):
    [b, a] = butter_highpass(min_f, fs, order=2)
    [b2, a2] = butter_lowpass(max_f, fs, order=2)
    x = lfilter(b, a, x)
    x = lfilter(b2, a2, x)
    return x
