import pickle
import os
import librosa
import numpy as np
import scipy
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt


class DataGeneratorPickles(Sequence):

    def __init__(self, filename, data_dir, set, batch_size=2800, type=np.float64):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """
        data = open(os.path.normpath('/'.join([data_dir, filename + '.pickle'])), 'rb')
        Z = pickle.load(data)
        y, keys, velocities, _, _, _, _ = Z[set]
        #if set == 'train':
        #    y_v, _, velocities_v, _, _, _, amps_v = Z['val']
        #    y = np.concatenate([y, y_v], axis=0)
        #    velocities = np.concatenate([velocities, velocities_v], axis=0)

        size = 1300
        w = scipy.signal.windows.tukey(4800, alpha=0.00005, sym=True)
        w2 = scipy.signal.windows.tukey(size, alpha=0.00005, sym=True)

        T = np.zeros((y.shape[0], y.shape[1]))
        self.T_s = np.zeros((y.shape[0], 4800))
        self.DCT = np.zeros((y.shape[0], 4800))
        self.DCT_s = np.zeros((y.shape[0], size))
        for i in range(y.shape[0]):
            D = librosa.stft(y[i])
            D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=10)
            Tistf = librosa.istft(D_percussive)
            T[i] = np.pad(Tistf,  [0, y.shape[1] - Tistf.shape[0]])

            self.T_s[i] = T[i, :4800] * w
            self.DCT[i] = scipy.fftpack.dct(self.T_s[i], type=2, norm='ortho')
            self.DCT_s[i] = self.DCT[i, :size] * w2

        self.filename = filename
        self.batch_size = batch_size

        self.velocities = velocities.reshape(-1, 1)/111

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.velocities.shape[0])

    def __len__(self):
        return int(self.velocities.shape[0]/self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        inputs = self.velocities[indices]
        targets = self.DCT_s[indices]

        return inputs, targets
