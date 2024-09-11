import pickle
import os
import librosa
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils import filterAudio


class DataGeneratorPickles(Sequence):

    def __init__(self, filename, data_dir, set, steps, model, batch_size=2800, type=np.float64):
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
        #    y_v, _, velocities_v, _, _, _, _ = Z['val']
        #    y = np.concatenate([y, y_v], axis=0)
        #    velocities = np.concatenate([velocities, velocities_v], axis=0)

        ### add
        N = np.zeros((y.shape[0], y.shape[1]))
        for i in range(y.shape[0]):
            D = librosa.stft(y[i])
            D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=1)
            D_noise = D - (D_harmonic + D_percussive)
            noise = librosa.istft(D_noise)
            noise = np.pad(np.array(noise, dtype=type), [0, len(y[0]) - len(noise)])
            N[i] = noise

        self.set = set
        self.filename = filename
        self.batch_size = batch_size
        self.steps = steps
        self.noise = np.array(N, dtype=type)
        self.noise = self.noise[:, :]/np.max(self.noise)
        #self.noise = self.noise[:, self.noise.shape[1]//2:]/np.max(self.noise)
        self.ratio = self.noise.shape[1] // (steps)

        lim = self.noise.shape[1]//self.steps*self.steps
        self.noise = self.noise[:, :lim]
        self.ratio = self.noise.shape[1] // (steps)

        self.velocities = velocities.reshape(-1, 1)/111
        self.n_note = self.velocities.shape[0]

        #########

        # waveforms
        self.k = np.arange(0, self.ratio).reshape(-1, 1)
        self.k = np.array(self.k, dtype=np.float32)
        self.k = np.repeat(self.k.T, self.n_note, axis=0).reshape(-1, 1)

        #self.y = self.y.reshape(-1, steps)
        self.N = self.noise.reshape(-1, steps)

        self.rms = np.abs(tf.reduce_mean(np.square(self.N), axis=-1)).reshape(-1, 1)
        self.mean = np.mean(self.N, axis=-1).reshape(-1, 1)

        self.velocities = np.repeat(self.velocities, self.ratio, axis=0).reshape(-1, 1)

        self.prev_v = None
        self.model = model
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

        if self.prev_v != self.velocities[indices[0], 0]:
            self.model.reset_states()

        self.prev_v = self.velocities[indices[0], 0]


        inputs = [self.k[indices], self.velocities[indices]]
        targets = {'output_1': self.rms[indices].reshape(self.batch_size, -1),
                   'output_2': self.N[indices].reshape(self.batch_size, -1),
                   'output_3': self.mean[indices].reshape(self.batch_size, -1)}

        return (inputs, (targets))
