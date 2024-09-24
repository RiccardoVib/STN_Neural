import pickle
import os
import scipy
from Utils import AttTime
import librosa
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt


class DataGeneratorPickles(Sequence):

    def __init__(self, filename, data_dir, set, steps, model, batch_size=2800, stage='', type=np.float64):
        """
        Initializes a data generator object
        :param filename: name of the file to load
        :param data_dir: the directory in which data are stored
        :param set: train or validation set [string]
        :param steps: number of timesteps generated per iteration [int]
        :param model: the model object [model class]
        :param stage: the name of the training phase [string]
        """

        # load the data
        data = open(os.path.normpath('/'.join([data_dir, filename + '.pickle'])), 'rb')
        Z = pickle.load(data)
        y, keys, velocities, partials, f0, B, _ = Z[set]
 
        # compute harmonic component
        S = np.zeros((y.shape[0], y.shape[1]))
        attackTimes = np.zeros((y.shape[0]))
        for i in range(y.shape[0]):
            D = librosa.stft(y[i])
            D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=10)
            harmonic = librosa.istft(D_harmonic)
            harmonic = np.pad(np.array(harmonic, dtype=type), [0, len(y[0]) - len(harmonic)])
            S[i] = harmonic
            attackTime = AttTime(S[i])
            attackTimes[i] = attackTime

        self.stage = stage
        self.filename = filename
        self.batch_size = batch_size
        self.minibatch_size = 1
        self.steps = steps
        self.y = np.array(y, dtype=type)
        w = scipy.signal.windows.tukey(self.batch_size, alpha=0.000005, sym=True).reshape(1,-1)
        self.S = np.array(S[:, :self.batch_size], dtype=type)*w
        self.ratio = self.S.shape[1] // (steps)

        # metadata
        self.f0 = f0.reshape(-1, 1)
        self.velocities = velocities.reshape(-1, 1)/111
        self.partials = partials.reshape(-1, 6)
        self.B = B
        self.attackTimes = np.array(attackTimes, dtype=type).reshape(-1, 1)
        self.n_note = self.velocities.shape[0]

        #########

        # indices
        self.k = np.arange(0, self.ratio).reshape(-1, 1)
        self.k = np.array(self.k, dtype=np.float32)
        self.k = np.repeat(self.k.T, self.n_note, axis=0).reshape(-1, self.minibatch_size, steps)

        # waveforms
        self.S = self.S.reshape(-1, self.minibatch_size, steps)

        self.rms = np.abs(tf.reduce_mean(np.square(self.S), axis=-1)).reshape(-1, minibatch_size, 1)
        self.alfas = np.max(np.abs(self.S), axis=-1).reshape(-1, minibatch_size, 1)

        self.f0 = np.repeat(self.f0, self.ratio*self.n_note, axis=0).reshape(-1, minibatch_size, 1)
        self.partials = np.repeat(self.partials, self.ratio*self.n_note, axis=0).reshape(-1, minibatch_size, 6)
        self.velocities = np.repeat(self.velocities, self.ratio, axis=0).reshape(-1, minibatch_size, 1)
        self.B = np.repeat(self.B, self.ratio*self.n_note, axis=0).reshape(-1, 1, minibatch_size, 1)
        self.attackTimes = np.repeat(self.attackTimes, self.ratio, axis=0).reshape(-1, minibatch_size, 1)

        self.prev = None
        self.prev_v = None
        self.model = model
        self.on_epoch_end()

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.f0.shape[0])

    def __len__(self):
        # compute the needed number of iterations before conclude one epoch
        return int(self.f0.shape[0]/(self.batch_size//self.minibatch_size))

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # get the indices of the requested batch
        indices = self.indices[idx * (self.batch_size//self.minibatch_size):(idx + 1) * (self.batch_size//self.minibatch_size)]

        #reset the states if velocity changes
        if self.prev != self.f0[indices[0], 0] or self.prev_v != self.velocities[indices[0], 0]:
            self.model.reset_states()

        self.prev = self.f0[indices[0], 0]
        self.prev_v = self.velocities[indices[0], 0]

        inputs = [self.f0[indices], self.k[indices], self.velocities[indices], self.B[indices], self.attackTimes[indices]]
        
        if self.stage == 'B':
            targets = {'output_1': self.partials[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, 6)}
    
        elif self.stage == 'S':
                targets = {'output_1': self.S[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, self.steps)}

        elif self.stage == 'A':
            targets = {
                   'output_1': self.S[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, self.steps),
                   'output_2': self.rms[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, 1),
                   'output_3': self.alfas[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, 1)
                   }
        else:
            targets = {'output_1': self.partials[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, 6),
                   'output_2': self.phase,
                   'output_3': self.S[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, self.steps),
                   'output_4': self.rms[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, 1),
                   'output_5': self.alfas[indices].reshape(self.batch_size//self.minibatch_size, self.minibatch_size, 1)
                      }

        return (inputs, (targets))
