import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class centLoss(tf.keras.losses.Loss):
    """ cent error """

    def __init__(self, delta=0., name="Cent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    def call(self, y_true, y_pred):

        loss = tf.keras.metrics.mean_squared_error(tf.experimental.numpy.log2(y_true+self.delta), tf.experimental.numpy.log2(y_pred+self.delta))

        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}


class STFT_loss(tf.keras.losses.Loss):
    """ multi-STFT error """

    def __init__(self, m=[32, 64, 128, 256], fft_size=2048, num_samples=1200, name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.fft_size = fft_size
        self.num_samples = num_samples
        self.delta = 1e-12#0.000001,
    def call(self, y_true, y_pred):

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])
        loss = 0
        pads = [[0, 0] for _ in range(2)]
        pad_amount = int((2048) // 2)  # Symmetric even padding like librosa.
        pads[1] = [pad_amount, pad_amount]
        y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred = K.abs(tf.signal.stft(y_pred, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            Y_true = K.pow(Y_true, 2)
            Y_pred = K.pow(Y_pred, 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1) + self.delta)
            loss += tf.divide(tf.norm((l_true - l_pred), ord=1), tf.norm(l_true, ord=1) + self.delta)
        
        return loss/len(self.m)

    def get_config(self):
        config = {
            'm': self.m,
            'fft_size': self.fft_size,
            'num_samples': self.num_samples
        }
        base_config = super().get_config()
        return {**base_config, **config}