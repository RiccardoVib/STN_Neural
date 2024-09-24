import tensorflow as tf
import math as m
import numpy as np


class InharmonicLayer(tf.keras.layers.Layer):
    def __init__(self, harmonics=32, batch_size=1, num_frames=1, Fs=24000, trainable=True, type=tf.float32):
        """
        Inharmonic layer
            :param harmonics: number of harmonics to compute
            :param batch_size: bacth size
            :param num_frames: number of frames per batch
            :param Fs: sampling rate
            :param trainable: if train the layers
        """
        super(InharmonicLayer, self).__init__()
        self.type = type
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.trainable = trainable
        self.n = tf.expand_dims(tf.linspace(1, harmonics, harmonics), axis=0)
        self.n = tf.repeat(self.n, self.batch_size, axis=0)
        self.n = tf.cast(self.n, dtype=self.type)
        self.ones = tf.ones((self.batch_size, harmonics))
        self.correction = tf.Variable(tf.zeros((1, 1, 1)), trainable=trainable)

    def __call__(self, freq_inputs, B_inputs):
        """
        out:
        partials: partials distribution
        final_n: the vector needed to compute the partials distribution
        """

        B = tf.reshape(B_inputs, [self.batch_size, 1, 1]) # starting from realistic value
        n2 = tf.pow(self.n, 2)  # BxH
        corr = self.correction

        corr = tf.clip_by_value(corr, 0., 10.) # clip if too big
        corr = tf.repeat(corr, self.batch_size, axis=0)
        Bn = tf.keras.layers.Multiply()([corr, B])
        Bn = tf.keras.layers.Multiply()([n2, Bn])  # BxH
        Bn = tf.keras.layers.Add()([self.ones, Bn])  # BxH

        final_n = tf.keras.layers.Multiply()([self.n, Bn])  # BxH
        final_n = tf.reshape(final_n, [self.batch_size // self.num_frames, self.num_frames, -1])
        partials = tf.keras.layers.Multiply()([final_n, freq_inputs])  # BxDxH

        return partials, final_n


class DeltaLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size=1, Fs=24000, trainable=True, num_frames=2400, type=tf.float32):
        """
        Detuning layer
            :param batch_size: bacth size
            :param num_frames: number of frames per batch
            :param Fs: sampling rate
            :param trainable: if train the layers
        """
        super(DeltaLayer, self).__init__()
        self.type = type
        self.num_frames = num_frames
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.batch_size = batch_size
        self.trainable = trainable

        self.linear_t2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(8, name='t2', input_shape=(self.batch_size, self.num_frames, 1),
                                  trainable=self.trainable,
                                  dtype=self.type))  # 128
        self.linear_t2_1 = tf.keras.layers.Dense(8, name='t2_1', activation='gelu',
                                                 input_shape=(self.batch_size, self.num_frames, 1),
                                                 trainable=self.trainable, dtype=self.type)
        # self.t2Norm = tf.keras.layers.BatchNormalization(trainable=self.trainable)
        self.t2Norm = tf.keras.layers.BatchNormalization()

        self.linearOut_t2 = tf.keras.layers.Dense(1, activation='tanh', name='t2out', trainable=self.trainable,
                                                  dtype=self.type)

    def __call__(self, vel_inputs):
        """
        out:
        t2: detuning value
        """
        t2 = self.linear_t2(vel_inputs)
        t2 = self.linear_t2_1(t2)
        t2 = self.t2Norm(t2)
        t2 = self.linearOut_t2(t2)

        return t2


class SinuosoidsGenerator(tf.keras.layers.Layer):
    def __init__(self, batch_size, harmonics=24, num_steps=24, num_frames=2400, Fs=24000, type=tf.float32):
        """
        Sinuosoids Generator
            :param harmonics: number of harmonics to compute
            :param batch_size: bacth size
            :param num_frames: number of frames per batch
            :param Fs: sampling rate
            :param num_steps: number of timesteps to compute per iteration
        """
        super(SinuosoidsGenerator, self).__init__()
        self.type = type
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.batch_size = batch_size
        self.harmonics = harmonics
        self.num_steps = num_steps
        self.num_frames = num_frames

    def __call__(self, freqs_inputs, index_inputs):
        """
        out:
        partials: the set of sine waves representing the partials,
        f: the related frequency bins
        t: the related time axis
        """
        f = tf.divide(freqs_inputs, self.Fs, name='divide')  # BxH
        x = tf.multiply(self.twopi, f, name='mul')  # BxH
        t = tf.ones((self.batch_size // self.num_frames, self.num_frames, self.num_steps, self.harmonics))  # BxTxH
        ti = tf.constant(np.arange(self.num_steps * self.num_frames).reshape(1, self.num_frames, self.num_steps, 1),
                         dtype=self.type)

        t = tf.multiply(t, ti)  # BxTxH

        k = tf.expand_dims(index_inputs, axis=-1)  # Bx1x1
        k = tf.multiply(k, self.num_steps)
        t = t + k[:, :1, :, :]
        x = tf.expand_dims(x, axis=2)  # Bx1xH
        partials = tf.multiply(x, t, name='mul2')  # BxTxH

        return partials, f, t


class DecayLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size, harmonics=24, Fs=24000, trainable=True, num_frames=2400, g=0, phan=False,
                 type=tf.float32):
        """
        Decay Layer
            :param harmonics: number of harmonics to compute
            :param batch_size: bacth size
            :param num_frames: number of frames per batch
            :param Fs: sampling rate
            :param g: starting mode for compute the longitudinal displacements
            :param phan: if include phantom partials
        """
        super(DecayLayer, self).__init__()
        self.batch_size = batch_size
        self.type = type
        self.num_frames = num_frames
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.harmonics = harmonics
        self.trainable = trainable
        self.p = 2
        self.units = self.harmonics * self.p
        self.g = g
        self.phan = phan

        self.alfaIn = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.units // 2, name='alfaIn',
                                  batch_input_shape=(self.batch_size // self.num_frames, self.num_frames, 2),
                                  trainable=self.trainable, dtype=self.type))  #
        self.alfa = tf.keras.layers.LSTM(self.units // 2, stateful=True, return_sequences=True, name='alfa',
                                         trainable=self.trainable, dtype=self.type)

        self.alfaDenseGelu = tf.keras.layers.Dense(self.units // 2, activation='gelu', name='alfa3',
                                                   trainable=self.trainable, dtype=self.type)  ##
        self.alfaOut = tf.keras.layers.Dense(self.units, name='alfaOut', trainable=self.trainable, dtype=self.type)
        self.alfaNorm = tf.keras.layers.BatchNormalization()
        self.alfaNormPre = tf.keras.layers.BatchNormalization()

        self.Att = tf.keras.layers.MultiHeadAttention(1, 8)  # , attention_axes=[2])

        self.b = tf.Variable(tf.random.normal([4]), trainable=self.trainable)
        self.perm = tf.keras.layers.Permute((2, 1))

    def compute_decay_rate(self, b0, b1, b2, b3, t_s): # compute the decay rate given the b coeficients
        decay_rate = tf.add(b0, b1)
        decay_rate = tf.add(decay_rate, b2)
        decay_rate = tf.add(decay_rate, b3)
        decay_rate = tf.abs(decay_rate)  # BxDxH
        decay_rate = tf.expand_dims(decay_rate, axis=2)  # Bx1xH
        decay_rate = tf.multiply(decay_rate, t_s)
        return decay_rate

    def compute_free_long(self, alfas): # compute the even longitudinal waves

        alfa_even = tf.pow(alfas[0][:, :, :, self.g:], 2.)
        alfa_even2 = tf.pow(alfas[1][:, :, :, self.g:], 2.)
        return [alfa_even, alfa_even2]

    def compute_phantoms(self, alfas): # compute the odd longitudinal waves
        alfa_odd = []
        alfa_odd2 = []
        alfa_odd3 = []

        for i in range(self.g, alfas[0].shape[-1] - 1):
            alfa_odd.append(tf.multiply(i * alfas[0][:, :, :, i], (i + 1) * alfas[0][:, :, :, i + 1]))
            alfa_odd.append(tf.multiply(i * alfas[0][:, :, :, i], (i + 1) * alfas[0][:, :, :, i + 1]))

            alfa_odd2.append(tf.multiply(i * alfas[1][:, :, :, i], (i + 1) * alfas[1][:, :, :, i + 1]))
            alfa_odd2.append(tf.multiply(i * alfas[1][:, :, :, i], (i + 1) * alfas[1][:, :, :, i + 1]))

            alfa_odd3.append(tf.multiply(i * alfas[0][:, :, :, i], (i + 1) * alfas[1][:, :, :, i + 1]))
            alfa_odd3.append(tf.multiply(i * alfas[0][:, :, :, i], (i + 1) * alfas[1][:, :, :, i + 1]))

        alfa_odd = tf.transpose(tf.convert_to_tensor(alfa_odd), perm=[1, 2, 3, 0])
        alfa_odd2 = tf.transpose(tf.convert_to_tensor(alfa_odd2), perm=[1, 2, 3, 0])
        alfa_odd3 = tf.transpose(tf.convert_to_tensor(alfa_odd3), perm=[1, 2, 3, 0])
        return [alfa_odd, alfa_odd2, alfa_odd3]

    def __call__(self, vel_inputs, f, t, K, partials, t_even, t_odd, attackTime):
        """
        out:
        decay_v: decay values for the vertical displacements
        decay_h: decay values for the horizontal displacements
        attackTime_t: attack time
        phantom_decays: decay values for the longitudinal displacements
        """
        attackTime_t = tf.multiply(attackTime, t[:, :, :, 0])  # BxTx1
        attackTime_t = tf.math.minimum(attackTime_t, 1.) # not used, for debug

        b_v = tf.abs(tf.nn.gelu(self.b))

        all_inp = tf.keras.layers.Concatenate(axis=-1)([vel_inputs, K])  # Bx3
        alfa_v = self.alfaIn(all_inp)  # Bx1xH
        alfa_v = self.alfaNormPre(alfa_v)  # BxDxH
        alfa_v = self.alfa(alfa_v)  # BxDxH

        alfa_v = self.alfaDenseGelu(alfa_v)  # BxDxH
        alfa_v = self.alfaNorm(alfa_v)  # BxDxH

        alfa_v = self.perm(alfa_v)  # BxHxD
        alfa_v = self.Att(alfa_v, alfa_v)
        alfa_v = self.perm(alfa_v)  # BxHxD

        alfa_v = self.alfaOut(alfa_v)  # BxDxH
        alfa_v = tf.abs(alfa_v)

        alfa_v = tf.clip_by_value(alfa_v, 0., 1.)

        alfa_v, alfa_h = tf.split(alfa_v, [self.harmonics, self.harmonics], axis=-1)

        b0, b1, b2, b3 = tf.split(b_v, 4, axis=-1)
        f3 = tf.pow(f[1], 3)
        t_s = tf.divide(t, self.Fs)
        f_sqrt = tf.sqrt(f[1])

        b3_ = tf.multiply(f[1], b3)  # BxDxH #####
        b2_ = tf.multiply(f3, b2)  # BxDxH #####
        b1_ = tf.multiply(b1, f_sqrt)

        alfa_h = tf.expand_dims(alfa_h, axis=2)  # Bx1xH
        decay_rates_h = self.compute_decay_rate(b0, b1_, b2_, b3_, t_s)
        decay_rates_h = tf.multiply(decay_rates_h, m.pi)
        decay_h = tf.math.exp(-decay_rates_h)
        decay_h = tf.multiply(decay_h, alfa_h)

        f3 = tf.pow(f[0], 3)
        f_sqrt = tf.sqrt(f[0])

        b3_ = tf.multiply(f[0], b3)  # BxH #####
        b2_ = tf.multiply(f3, b2)  # BxH #####
        b1_ = tf.multiply(b1, f_sqrt)  # BxH #####

        alfa_v = tf.expand_dims(alfa_v, axis=2)  # Bx1xH
        decay_rates_v = self.compute_decay_rate(b0, b1_, b2_, b3_, t_s)  # BxTxH #####
        decay_rates_v = tf.multiply(decay_rates_v, m.pi)
        decay_v = tf.math.exp(-decay_rates_v)
        decay_v = tf.multiply(decay_v, alfa_v)

        if self.phan:
            ## free resposne
            f_even = tf.divide(partials[0], self.Fs, name='divide')  # BxH
            f_even2 = tf.divide(partials[1], self.Fs, name='divide')  # BxH
            f_odd = tf.divide(partials[2], self.Fs, name='divide')  # BxH
            f_odd2 = tf.divide(partials[3], self.Fs, name='divide')  # BxH
            f_odd3 = tf.divide(partials[4], self.Fs, name='divide')  # BxH

            alfas_even = self.compute_free_long([decay_v, decay_h])  # BxTxH #####

            f3 = tf.pow(f_even, 3)
            f_sqrt = tf.sqrt(f_even)
            b3_ = tf.multiply(f_even, b3)  # BxH #####
            b2_ = tf.multiply(f3, b2)  # BxH #####
            b1_ = tf.multiply(b1, f_sqrt)  # BxH #####
            decay_rates_even = self.compute_decay_rate(b0, b1_, b2_, b3_, t_even)
            decay_rates_even = tf.multiply(decay_rates_even, m.pi)
            decay_even = tf.math.exp(-decay_rates_even)
            decay_even = tf.multiply(decay_even, alfas_even[0])

            f3 = tf.pow(f_even2, 3)
            f_sqrt = tf.sqrt(f_even2)
            b3_ = tf.multiply(f_even2, b3)  # BxH #####
            b2_ = tf.multiply(f3, b2)  # BxH #####
            b1_ = tf.multiply(b1, f_sqrt)  # BxH #####
            decay_rates_even2 = self.compute_decay_rate(b0, b1_, b2_, b3_, t_even)
            decay_rates_even2 = tf.multiply(decay_rates_even2, m.pi)
            decay_even2 = tf.math.exp(-decay_rates_even2)
            decay_even2 = tf.multiply(decay_even2, alfas_even[1])

            ## coupling
            alfas_odd = self.compute_phantoms([decay_v, decay_h])  # BxTxH #####

            f3 = tf.pow(f_odd, 3)
            f_sqrt = tf.sqrt(f_odd)
            b3_ = tf.multiply(f_odd, b3)  # BxH #####
            b2_ = tf.multiply(f3, b2)  # BxH #####
            b1_ = tf.multiply(b1, f_sqrt)  # BxH #####
            decay_rates_odd = self.compute_decay_rate(b0, b1_, b2_, b3_, t_odd)
            decay_rates_odd = tf.multiply(decay_rates_odd, m.pi)
            decay_odd = tf.math.exp(-decay_rates_odd)
            decay_odd = tf.multiply(decay_odd, alfas_odd[0])

            f3 = tf.pow(f_odd2, 3)
            f_sqrt = tf.sqrt(f_odd2)
            b3_ = tf.multiply(f_odd2, b3)  # BxH #####
            b2_ = tf.multiply(f3, b2)  # BxH #####
            b1_ = tf.multiply(b1, f_sqrt)  # BxH #####
            decay_rates_odd2 = self.compute_decay_rate(b0, b1_, b2_, b3_, t_odd)
            decay_rates_odd2 = tf.multiply(decay_rates_odd2, m.pi)
            decay_odd2 = tf.math.exp(-decay_rates_odd2)
            decay_odd2 = tf.multiply(decay_odd2, alfas_odd[1])

            f3 = tf.pow(f_odd3, 3)
            f_sqrt = tf.sqrt(f_odd3)
            b3_ = tf.multiply(f_odd3, b3)  # BxH #####
            b2_ = tf.multiply(f3, b2)  # BxH #####
            b1_ = tf.multiply(b1, f_sqrt)  # BxH #####
            decay_rates_odd3 = self.compute_decay_rate(b0, b1_, b2_, b3_, t_odd)
            decay_rates_odd3 = tf.multiply(decay_rates_odd3, m.pi)
            decay_odd3 = tf.math.exp(-decay_rates_odd3)
            decay_odd3 = tf.multiply(decay_odd3, alfas_odd[2])

            phantom_decays = 0.1 * [decay_even, decay_even2, decay_odd, decay_odd2, decay_odd3]

            return decay_v, decay_h, attackTime_t, phantom_decays  # BxTxH
        else:

            return decay_v, decay_h, attackTime_t
