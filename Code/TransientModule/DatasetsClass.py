import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import scipy

class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, set, size, model, batch_size=2800, type=np.float64):

        data = open(os.path.normpath('/'.join([data_dir, 'DatasetSingleNote_split.pickle'])), 'rb')
        Z = pickle.load(data)
        y, keys, velocities, partials, f0, B, S, T, N, DCT, attack = Z[set]

        # waveforms
        self.T = np.array(T, dtype=type)
        w = scipy.signal.windows.tukey(4800, alpha=0.05, sym=True)
        self.T_short = T[:, :4800] * w
        self.size = size
        self.set = set
        T_s = []
        for i in range(self.T_short.shape[0]):
            T_s.append(scipy.signal.resample_poly(self.T_short[i], 1, 4))
        self.T_s = np.array(T_s)

        self.dct = scipy.fftpack.dct(self.T_s, type=2, norm='ortho')
        self.w = scipy.signal.windows.tukey(self.size, alpha=0.05, sym=True)
        self.dct = self.dct[:, :self.size] * w

        ###metadata
        self.f0 = f0.reshape(-1, 1) / np.max(f0)
        self.velocities = velocities.reshape(-1, 1) / np.max(velocities)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.f0.shape[0])

    def __len__(self):
        return int(self.f0.shape[0]/self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        inputs = [self.f0[indices], self.velocities[indices]]
        targets = self.dct[indices].reshape(self.batch_size, -1)

        return (inputs, (targets))
