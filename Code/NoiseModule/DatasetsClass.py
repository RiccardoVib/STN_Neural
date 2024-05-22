import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt

class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, set, steps, model, batch_size=1, type=np.float64):

        data = open(os.path.normpath('/'.join([data_dir, 'DatasetSingleNote_split.pickle'])), 'rb')
        Z = pickle.load(data)

        y, keys, velocities, partials, f0, Bs, S, T, N, DCT, attackTimes = Z[set]
        
        self.N = np.array(N, dtype=type)/np.max(N)
        self.N = self.N[:, 12000:]
        self.N = self.N[:, len(self.N[0])//2:]
        self.N = self.N[:, :140000]
                
        self.ratio = self.N.shape[1] // steps
        self.N = self.N.reshape(-1, steps)
        self.batch_size = batch_size

        ###metadata
        self.f0 = f0.reshape(-1, 1)/np.max(f0)
        self.n_note = f0.shape[0]
        self.velocities = velocities / np.max(velocities)
        self.velocities = self.velocities.reshape(-1, 1)
        self.set = set
        #########

        self.k = np.arange(0, self.ratio).reshape(-1, 1)
        self.k = np.array(self.k, dtype=np.float32)
        self.k = np.repeat(self.k.T, self.n_note, axis=0).reshape(-1, 1)
        self.rms = np.abs(tf.reduce_mean(np.square(self.N), axis=-1)).reshape(-1, 1)
        self.mean = np.mean(self.N, axis=-1).reshape(-1, 1)
        self.f0 = np.repeat(self.f0, self.ratio, axis=0).reshape(-1, 1)
        self.velocities = np.repeat(self.velocities, self.ratio, axis=0).reshape(-1, 1)
        self.steps = steps
        self.prev = None
        self.prev_v = None
        self.model = model
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
        if self.prev != self.f0[indices[0]] or self.prev_v != self.velocities[indices[0]]:
            self.model.reset_states()

        self.prev = self.f0[indices[0]]
        self.prev_v = self.velocities[indices[0]]

        inputs = [self.f0[indices], self.k[indices], self.velocities[indices]]
        targets = {'output_1': self.rms[indices].reshape(self.batch_size, -1), 'output_2': self.N[indices].reshape(self.batch_size, -1), 'output_3': self.mean[indices].reshape(self.batch_size, -1)}

        return (inputs, (targets))
