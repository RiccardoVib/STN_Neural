import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence

class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, mini_batch_size, input_size, set, model, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param filename: the name of the dataset
          :param input_size: the inpput size
          :param cond: the number of conditioning values
          :param batch_size: The size of each batch returned by __getitem__
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.input_size = input_size

        # prepare the input, taget and conditioning matrix
        self.x, self.y, self.f0, self.velocities, self.k, lim = self.prepareXYZ(data_dir, set)

        self.max_1 = (self.x.shape[1] // self.mini_batch_size) - 1
        self.max = (self.max_1 // self.batch_size) - 1

        self.training_steps = self.max

        self.model = model
        self.prev = None
        self.prev_v = None

        self.on_epoch_end()
        
    def prepareXYZ(self, data_dir, set):

        # load all the audio files
        
        data = open(os.path.normpath('/'.join([data_dir, 'DiskChordUpright_split.pickle'])), 'rb')
        Z = pickle.load(data)
        y, y_sum, keys, velocities, f0, S, T, N, DCT, S_sum, T_sum, N_sum, DCT_sum = Z[set]
        
        x = np.array(y_sum, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        ###metadata
        f0 = f0.reshape(-1, 1, 3) / np.max(f0)
        n_note = f0.shape[0]
        velocities = velocities.reshape(-1, 1, 1) / np.max(velocities)
        ratio = y.shape[1]

        k = np.arange(0, ratio).reshape(1, -1, 1)
        k = np.array(k, dtype=np.float32)
        k = np.repeat(k, n_note, axis=0)/ np.max(k)

        f0 = np.repeat(f0, ratio, axis=1)
        velocities = np.repeat(velocities, ratio, axis=1)

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        f0 = f0.reshape(1, -1, 3)
        velocities = velocities.reshape(1, -1)
        k = k.reshape(1, -1)

        # how many iteration it is needed
        N = int((x.shape[1]/self.mini_batch_size) / self.batch_size)-1
        # how many total samples is the audio
        lim = int(N * self.batch_size) * self.mini_batch_size - 1
        x = x[:, :lim]
        y = y[:, :lim]
        f0 = f0[:, :lim, :]
        velocities = velocities[:, :lim]
        k = k[:, :lim]

        return x, y, f0, velocities, k, lim

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(0, self.x.shape[1])

    def __len__(self):
        # compute the itneeded number of iteration before conclude one epoch
        return int(self.max)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        # get the indices of the requested batch
        indices = self.indices[idx*self.mini_batch_size*self.batch_size:(idx+1)*self.mini_batch_size*self.batch_size]

        if self.prev != self.f0[0, indices[0], 0] or self.prev_v != self.velocities[0, indices[0]]:
            self.model.reset_states()

        self.prev = self.f0[0, indices[0], 0]
        self.prev_v = self.velocities[0, indices[0]]

        # fill the batches
        X = np.array(self.x[0, indices]).reshape(self.batch_size, self.mini_batch_size, 1)
        Y = np.array(self.y[0, indices]).reshape(self.batch_size, self.mini_batch_size, 1)
        F0 = np.array(self.f0[0, indices]).reshape(self.batch_size, self.mini_batch_size, 3)
        V = np.array(self.velocities[0, indices]).reshape(self.batch_size, self.mini_batch_size, 1)
        K = np.array(self.k[0, indices]).reshape(self.batch_size, self.mini_batch_size, 1)
        #rms = np.abs(tf.reduce_mean(np.square(Y), axis=1))

        inputs = [F0, V, K, X]
        targets = Y#, rms]

        return (inputs, (targets))