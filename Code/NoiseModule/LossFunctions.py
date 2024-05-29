import tensorflow as tf
from tensorflow.keras import backend as K


class STFT_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512, 1024], name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true = tf.reshape(y_true, [-1, self.m[0]])
        y_pred = tf.reshape(y_pred, [-1, self.m[0]])
        loss = 0
        for i in range(len(self.m)):

            pad_amount = int((self.m[i]-self.m[0])//2)  # Symmetric even padding like librosa.
            pads = [[0, 0] for _ in range(len(y_true.shape))]
            pads[1] = [pad_amount, pad_amount]
            y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
            y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

            Y_true = K.abs(tf.signal.stft(y_true, frame_length=self.m[i], frame_step=self.m[i] // 4, fft_length=self.m[i], pad_end=False))
            Y_pred = K.abs(tf.signal.stft(y_pred, frame_length=self.m[i], frame_step=self.m[i] // 4, fft_length=self.m[i], pad_end=False))

            Y_true = K.pow(Y_true, 2)
            Y_pred = K.pow(Y_pred, 2)

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
