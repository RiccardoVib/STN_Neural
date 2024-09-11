import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

class NMSELoss(tf.keras.losses.Loss):
    def __init__(self, delta=0., name="NMSE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    def call(self, y_true, y_pred):

        loss = tf.divide(tf.keras.metrics.mean_squared_error(y_true, y_pred), tf.norm(y_true + 1e-9, ord=1))
        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}

class MFCC_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256], fft_size=2048, num_samples=1200, name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.fft_size = fft_size
        self.num_samples = num_samples

    def call(self, y_true, y_pred):

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])
        loss = 0
        pads = [[0, 0] for _ in range(2)]
        pad_amount = int((self.fft_size - self.num_samples) // 2)  # Symmetric even padding like librosa.
        pads[1] = [pad_amount, pad_amount]
        y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        sample_rate = 24000

        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred = K.abs(tf.signal.stft(y_pred, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            num_spectrogram_bins = Y_true.shape[-1]
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                upper_edge_hertz)
            Y_true = tf.tensordot(
                Y_true, linear_to_mel_weight_matrix, 1)
            Y_true.set_shape(Y_true.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            Y_true = tf.math.log(Y_true + 1e-6)
            # Compute MFCCs from log_mel_spectrograms and take the first 13.
            Y_true = tf.signal.mfccs_from_log_mel_spectrograms(Y_true)  # [..., :13]

            num_spectrogram_bins = Y_pred.shape[-1]
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                upper_edge_hertz)
            Y_pred = tf.tensordot(
                Y_pred, linear_to_mel_weight_matrix, 1)
            Y_pred.set_shape(Y_pred.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            Y_pred = tf.math.log(Y_pred + 1e-6)
            # Compute MFCCs from log_mel_spectrograms and take the first 13.
            Y_pred = tf.signal.mfccs_from_log_mel_spectrograms(Y_pred)  # [..., :13]

            #l_true = K.log(Y_true + 1)
            #l_pred = K.log(Y_pred + 1)

            #loss += tf.divide(tf.norm((l_true - l_pred), ord=1), tf.norm(l_true + 0.000001, ord=1))
            loss += tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true + 0.000001, ord=1))

        return loss/len(self.m)

    def get_config(self):
        config = {
            'm': self.m,
            'fft_size': self.fft_size,
            'num_samples': self.num_samples
        }
        base_config = super().get_config()
        return {**base_config, **config}
        
class centLoss(tf.keras.losses.Loss):
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



class phase_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512, 1024], fft_size=2048, num_samples=1200, name="phase", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.fft_size = fft_size
        self.num_samples = num_samples

    def call(self, y_true, y_pred):

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])
        loss = 0
        pads = [[0, 0] for _ in range(2)]
        pad_amount = int((self.fft_size - self.num_samples) // 2)  # Symmetric even padding like librosa.
        pads[1] = [pad_amount, pad_amount]
        y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

        for i in range(len(self.m)):

            stft_t = tf.signal.stft(y_true, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True)
            stft_p = tf.signal.stft(y_pred, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True)
            Y_true = (tf.math.imag(stft_t))
            Y_pred = (tf.math.imag(stft_p))
            #r_t = (tf.math.real(stft_t))
            #r_p = (tf.math.real(stft_p))
            #phase = tf.math.atan2(phase)

            #Y_true = tf.math.atan2(r_t, i_t)
            #Y_pred = tf.math.atan2(r_p, i_p)

            #loss += tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1))
            loss += tf.norm((Y_true - Y_pred), ord=1)

        return loss / len(self.m)

    def get_config(self):
        config = {
            'm': self.m,
            'fft_size': self.fft_size,
            'num_samples': self.num_samples
        }
        base_config = super().get_config()
        return {**base_config, **config}
