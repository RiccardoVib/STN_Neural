import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, set, steps, model, batch_size=2800, type=np.float64):
        """
        Initializes a data generator object
        :param data_dir: the directory in which data are stored
          :param set: train or validation set [string]
          :param steps: number of timesteps generated per iteration [int]
          :param model: the model object [model class]
          :param batch_size: The size of each batch returned by __getitem__ [int]
        """
        # load the data
        data = open(os.path.normpath('/'.join([data_dir, 'DiskChordUpright_split.pickle'])), 'rb')
        Z = pickle.load(data)
        y, y_sum, keys, velocities, f0, S, T, N, DCT, S_sum, T_sum, N_sum, DCT_sum = Z[set]

        #harmonic
        self.S = np.array(y, dtype=type)#336000
        self.S_sum = np.array(y_sum, dtype=type)#336000

        ###metadata
        self.f0 = f0.reshape(-1, 3) / np.max(f0)
        self.n_note = self.f0.shape[0]
        self.velocities = velocities.reshape(-1, 1) / np.max(velocities)
        self.ratio = y.shape[1] // steps

        #indices
        self.k = np.arange(0, self.ratio).reshape(-1, 1)
        self.k = np.array(self.k, dtype=np.float32)
        self.k = np.repeat(self.k.T, self.n_note, axis=0).reshape(-1, 1)

        self.S = self.S.reshape(-1, steps)
        self.S_sum = self.S_sum.reshape(-1, steps)
        self.rms = np.abs(tf.reduce_mean(np.square(self.S), axis=-1)).reshape(-1, 1)

        self.f0 = np.repeat(self.f0, self.ratio, axis=0)
        self.velocities = np.repeat(self.velocities, self.ratio, axis=0).reshape(-1, 1)
        self.batch_size = batch_size
        self.model = model
        self.prev = None
        self.prev_v = None
        self.on_epoch_end()

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.f0.shape[0])

    def __len__(self):
        # compute the needed number of iterations before conclude one epoch
        return int(self.f0.shape[0]/self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        #reset the states if velocity changes
        if self.prev != self.f0[indices[0], 0] or self.prev_v != self.velocities[indices[0]]:
            self.model.reset_states()

        self.prev = self.f0[indices[0], 0]
        self.prev_v = self.velocities[indices[0]]

        inputs = [self.f0[indices], self.velocities[indices], self.k[indices], self.S_sum[indices].reshape(self.batch_size, -1)]
        targets = {'output_1': self.S[indices].reshape(self.batch_size, -1), 'output_2': self.rms[indices].reshape(self.batch_size, -1)}
        return (inputs, (targets))
