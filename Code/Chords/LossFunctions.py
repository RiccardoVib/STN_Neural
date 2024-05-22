import tensorflow as tf
from tensorflow.keras import backend as K

class PSD_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512], name="PSD", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        loss = 0
        for i in range(len(self.m)):
            pad_amount = int(self.m[i] // 2)
            #pads = [[0, 0] for _ in range(len(y_true.shape))]
            #pads[0] = [pad_amount, pad_amount]
            pads = [[pad_amount, pad_amount]]
            y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
            y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

            Y_true = K.abs(tf.signal.rfft(y_true))
            Y_pred = K.abs(tf.signal.rfft(y_pred))
            Y_true = K.pow(Y_true, 2)
            Y_pred = K.pow(Y_pred, 2)

            #l_true = K.log(Y_true + 0.0001)
            #l_pred = K.log(Y_pred + 0.0001)

            loss += tf.norm((Y_true - Y_pred), ord=1)#/Y_true.shape[0]
            #loss += tf.norm((l_true - l_pred), ord=1)/Y_pred.shape[0]
        return loss/len(self.m)

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}

class STFT_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512, 1024], name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        loss = 0
        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            Y_true = K.pow(K.abs(Y_true), 2)
            Y_pred = K.pow(K.abs(Y_pred), 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.norm((l_true - l_pred), ord=1) + tf.norm((Y_true - Y_pred), ord=1)

        return loss/len(self.m)

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}