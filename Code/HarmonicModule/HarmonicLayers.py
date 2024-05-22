import tensorflow as tf
import math as m
import numpy as np

class EnhancementLayer(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, bias=True, dim=-1, trainable=True, type=tf.float32):
        super(EnhancementLayer, self).__init__()
        self.bias = bias
        self.steps = steps
        self.dim = dim
        self.trainable = trainable
        self.type = type
        self.b_size = b_size

        act='tanh'
        self.proj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps))
 
        self.conv = tf.keras.layers.Conv1D(16, 2, activation=act)
        self.conv2 = tf.keras.layers.Conv1D(8, 2, activation=act)
        self.conv3 = tf.keras.layers.Conv1D(4, 2, activation=act)
        self.conv3 = tf.keras.layers.Conv1D(1, 2, activation=act)

        self.out = tf.keras.layers.Dense(steps)
        
    def call(self, x):
        x = self.proj(x)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.squeeze(x, axis=-1)
        x = self.out(x)
        return x

class InharmonicLayer(tf.keras.layers.Layer):
    def __init__(self, harmonics=32, B=1, Fs=24000, trainable=True, type=tf.float32):
        super(InharmonicLayer, self).__init__()
        self.type = type
        self.max_ = tf.constant(494.62890625, dtype=self.type)
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.B = B
        self.trainable = trainable
        self.n = tf.expand_dims(tf.linspace(1, harmonics, harmonics), axis=0)
        self.n = tf.repeat(self.n, self.B, axis=0)
        self.n = tf.cast(self.n, dtype=self.type)
        self.ones = tf.ones((self.B, harmonics))

        self.linear = tf.keras.layers.Dense(8, name='B', trainable=self.trainable, dtype=self.type)
        self.linearOut = tf.keras.layers.Dense(1, activation='sigmoid', name='Bout', trainable=self.trainable, dtype=self.type)

    def __call__(self, freq_inputs):
        f_n = tf.divide(freq_inputs, self.max_)  # BxD

        B = self.linear(f_n)  # BxU
        B = self.linearOut(B)  # Bx1

        #####law3
        n2 = tf.pow(self.n, 2)  # BxH
        Bn = tf.keras.layers.Multiply()([n2, B])  # BxH
        Bn = tf.keras.layers.Add()([self.ones, Bn])  # BxH

        final_n = tf.keras.layers.Multiply()([self.n, Bn])  # BxH
        freqs_inputs = tf.keras.layers.Multiply()([final_n, freq_inputs])  # BxDxH

        return freqs_inputs, f_n, B, final_n


class DeltaLayer(tf.keras.layers.Layer):
    def __init__(self, B=1, Fs=24000, trainable=True, mul=1, type=tf.float32):
        super(DeltaLayer, self).__init__()
        self.type = type
        self.max_ = tf.constant(494.62890625, dtype=self.type)
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.B = B
        self.trainable = trainable

        self.linear_t2 = tf.keras.layers.Dense(8*mul, name='t2', input_shape=(self.B, 2), trainable=self.trainable, dtype=self.type)  # 128
        self.linearOut_t2 = tf.keras.layers.Dense(1, activation='tanh', name='t2out', trainable=self.trainable, dtype=self.type)

    def __call__(self, f_n, vel_inputs):
        all_inp = tf.keras.layers.Concatenate(axis=-1)([f_n, vel_inputs])  # Bx2

        t2 = self.linear_t2(all_inp)
        t2 = self.linearOut_t2(t2)

        return t2



class SinuosoidsGenerator(tf.keras.layers.Layer):
    def __init__(self, B, harmonics=24, num_steps=24, Fs=24000, type=tf.float32):
        """
        Time Varying feedforward delay line

        Args:
            max_delay (int, optional): Maximum length of delay line in samples. Defaults to 40000.
            channels (int, optional): Number of channels or audio. Defaults to 1.
        """
        super(SinuosoidsGenerator, self).__init__()
        self.type = type
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.B = B
        self.harmonics = harmonics
        self.num_steps = num_steps

    def __call__(self, freqs_inputs, index_inputs):
        ##### partials
        f = tf.divide(freqs_inputs, self.Fs, name='divide')  # BxH
        x = tf.multiply(self.twopi, f, name='mul')  # BxH
        t = tf.ones((self.B, self.num_steps, self.harmonics))  # BxTxH
        ti = tf.constant(np.arange(self.num_steps).reshape(1, self.num_steps, 1), dtype=self.type)
        t = tf.multiply(t, ti)  # BxTxH

        k = tf.expand_dims(index_inputs, axis=-1)  # Bx1x1
        k = tf.multiply(k, self.num_steps)
        t = t + k
        x = tf.expand_dims(x, axis=1)  # Bx1xH
        partials = tf.multiply(x, t, name='mul2')  # BxTxH

        return partials, f, t


class DecayLayer(tf.keras.layers.Layer):
    def __init__(self, B, harmonics=24, Fs=24000, trainable=True, mul=1, type=tf.float32):
        """
        Time Varying feedforward delay line

        Args:
            max_delay (int, optional): Maximum length of delay line in samples. Defaults to 40000.
            channels (int, optional): Number of channels or audio. Defaults to 1.
        """
        super(DecayLayer, self).__init__()
        self.B = B
        self.type = type
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.harmonics = harmonics
        self.trainable = trainable
        self.p = 2

        self.alfaIn = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8*mul, name='alfaIn', batch_input_shape=(self.B, 3), trainable=self.trainable, dtype=self.type))  # #128
        self.alfa = tf.keras.layers.LSTM(16*mul, stateful=True, return_sequences=False, name='alfa', trainable=self.trainable, dtype=self.type)  # 192 # 64
        self.alfaOut = tf.keras.layers.Dense(self.harmonics * self.p, name='alfaOut', activation=None, trainable=self.trainable, dtype=self.type)
        self.alfaNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.attackTime1 = tf.keras.layers.Dense(8, name='attackTime1', trainable=self.trainable, dtype=self.type)  # 128
        self.attackTime2 = tf.keras.layers.Dense(1, name='attackTime2', trainable=self.trainable, dtype=self.type)


        self.bIn = tf.keras.layers.Dense(8*mul, name='bIn', batch_input_shape=(self.B, 3), trainable=self.trainable, dtype=self.type)  # 128
        self.b = tf.keras.layers.Dense(16*mul, name='b', trainable=self.trainable, dtype=self.type)  # 192 #64
        self.bOut = tf.keras.layers.Dense(8, name='bOut', activation=None, trainable=self.trainable, dtype=self.type)
        self.bNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def compute_decay_rate(self, b0, b1, b2, b3, t_s):
        decay_rate = tf.add(b0, b1)
        decay_rate = tf.add(decay_rate, b2)
        decay_rate = tf.add(decay_rate, b3)
        decay_rate = tf.abs(decay_rate)  # BxDxH
        decay_rate = tf.expand_dims(decay_rate, axis=1)  # Bx1xH
        decay_rate = tf.multiply(decay_rate, t_s)
        return decay_rate

    def compute_free_long(self, alfas):
        g = 0#10
        alfa_even = tf.pow(alfas[0][:, :, g:], 2.)
        alfa_even2 = tf.pow(alfas[1][:, :, g:], 2.)
        return [alfa_even, alfa_even2]

    def compute_phantoms(self, alfas):
        alfa_odd = []
        alfa_odd2 = []
        alfa_odd3 = []
        g = 0#9
        for i in range(g, alfas[0].shape[-1]-1):
            alfa_odd.append(tf.multiply(i * alfas[0][:, :, i], (i+1) * alfas[0][:, :, i + 1]))
            alfa_odd.append(tf.multiply(i * alfas[0][:, :, i], (i+1) * alfas[0][:, :, i + 1]))

            alfa_odd2.append(tf.multiply(i * alfas[1][:, :, i], (i+1) * alfas[1][:, :, i + 1]))
            alfa_odd2.append(tf.multiply(i * alfas[1][:, :, i], (i+1) * alfas[1][:, :, i + 1]))

            alfa_odd3.append(tf.multiply(i * alfas[0][:, :, i], (i + 1) * alfas[1][:, :, i + 1]))
            alfa_odd3.append(tf.multiply(i * alfas[0][:, :, i], (i + 1) * alfas[1][:, :, i + 1]))

        alfa_odd = tf.transpose(tf.convert_to_tensor(alfa_odd), perm=[1, 2, 0])
        alfa_odd2 = tf.transpose(tf.convert_to_tensor(alfa_odd2), perm=[1, 2, 0])
        alfa_odd3 = tf.transpose(tf.convert_to_tensor(alfa_odd3), perm=[1, 2, 0])
        return [alfa_odd, alfa_odd2, alfa_odd3]

    def __call__(self, f_n, vel_inputs, f, t, K, partials, t_even, t_odd):
        all_inp = tf.keras.layers.Concatenate(axis=-1)([f_n, vel_inputs])  # BxDx2

        attackTime_s = self.attackTime1(all_inp)  # Bx2
        attackTime_s = self.attackTime2(attackTime_s)  # Bx1
        attackTime_s = tf.nn.sigmoid(attackTime_s)  # Bx1
        attackTime_s = tf.expand_dims(attackTime_s, axis=1)  # Bx1x1
        attackTime_t = tf.multiply(attackTime_s, t)  # BxTx1
        attackTime_t = tf.math.minimum(attackTime_t, 1.)

        all_inp = tf.keras.layers.Concatenate(axis=-1)([all_inp, K])  # Bx3
        all_inp = tf.expand_dims(all_inp, axis=-1)

        alfa_v = self.alfaIn(all_inp)  # Bx1xH
        alfa_v = self.alfa(alfa_v)  # BxDxH
        alfa_v = self.alfaNorm(alfa_v)  # BxDxH
        alfa_v = self.alfaOut(alfa_v)  # BxDxH

        b_v = self.bIn(all_inp[:, :, 0])  # BxH
        b_v = self.b(b_v)  # BxH
        b_v = self.bNorm(b_v)  # BxH
        b_v = self.bOut(b_v)  # BxH
        b_v = tf.nn.relu(b_v)

        alfa_v = tf.nn.sigmoid(alfa_v)  # BxDxH
        alfa_v = tf.divide(alfa_v, self.harmonics * 2)

        alfa_v, alfa_h = tf.split(alfa_v, [self.harmonics, self.harmonics], axis=-1)

        b_v, b_h = tf.split(b_v, [4, 4], axis=-1)

        b4, b5, b6, b7 = tf.split(b_h, 4, axis=-1)

        f3 = tf.pow(f[1], 3)
        t_s = tf.divide(t, self.Fs)
        f_sqrt = tf.sqrt(f[1])

        b7 = tf.multiply(f[1], b7)  # BxDxH #####
        b6 = tf.multiply(f3, b6)  # BxDxH #####
        b5 = tf.multiply(b5, f_sqrt)

        alfa_h = tf.expand_dims(alfa_h, axis=1)  # Bx1xH
        decay_rates_h = self.compute_decay_rate(b4, b5, b6, b7, t_s)
        decay_rates_h = tf.multiply(decay_rates_h, m.pi)
        decay_h = tf.math.exp(-decay_rates_h)
        decay_h = tf.multiply(decay_h, alfa_h)

        b0, b1, b2, b3 = tf.split(b_v, 4, axis=-1)

        f3 = tf.pow(f[0], 3)
        f_sqrt = tf.sqrt(f[0])

        b3 = tf.multiply(f[0], b3)  # BxH #####
        b2 = tf.multiply(f3, b2)  # BxH #####
        b1 = tf.multiply(b1, f_sqrt)  # BxH #####

        alfa_v = tf.expand_dims(alfa_v, axis=1)  # Bx1xH
        decay_rates_v = self.compute_decay_rate(b0, b1, b2, b3, t_s)  # BxTxH #####
        decay_rates_v = tf.multiply(decay_rates_v, m.pi)
        decay_v = tf.math.exp(-decay_rates_v)
        decay_v = tf.multiply(decay_v, alfa_v)
        #decay_v = tf.multiply(attackTime_t, decay_v)
        #decay_h = tf.multiply(attackTime_t, decay_h)

        ## free resposne
        f_even = tf.divide(partials[0], self.Fs, name='divide')  # BxH
        f_even2 = tf.divide(partials[1], self.Fs, name='divide')  # BxH
        f_odd = tf.divide(partials[2], self.Fs, name='divide')  # BxH
        f_odd2 = tf.divide(partials[3], self.Fs, name='divide')  # BxH
        f_odd3 = tf.divide(partials[4], self.Fs, name='divide')  # BxH

        alfas_even = self.compute_free_long([alfa_v, alfa_h])  # BxTxH #####

        b0, b1, b2, b3 = tf.split(b_v, 4, axis=-1)
        f3 = tf.pow(f_even, 3)
        f_sqrt = tf.sqrt(f_even)
        b3 = tf.multiply(f_even, b3)  # BxH #####
        b2 = tf.multiply(f3, b2)  # BxH #####
        b1 = tf.multiply(b1, f_sqrt)  # BxH #####
        decay_rates_even = self.compute_decay_rate(b0, b1, b2, b3, t_even)
        decay_rates_even = tf.multiply(decay_rates_even, m.pi)
        decay_even = tf.math.exp(-decay_rates_even)
        decay_even = tf.multiply(decay_even, alfas_even[0])

        b0, b1, b2, b3 = tf.split(b_h, 4, axis=-1)
        f3 = tf.pow(f_even2, 3)
        f_sqrt = tf.sqrt(f_even2)
        b3 = tf.multiply(f_even2, b3)  # BxH #####
        b2 = tf.multiply(f3, b2)  # BxH #####
        b1 = tf.multiply(b1, f_sqrt)  # BxH #####
        decay_rates_even2 = self.compute_decay_rate(b0, b1, b2, b3, t_even)
        decay_rates_even2 = tf.multiply(decay_rates_even2, m.pi)
        decay_even2 = tf.math.exp(-decay_rates_even2)
        decay_even2 = tf.multiply(decay_even2, alfas_even[1])

        ## coupling
        alfas_odd = self.compute_phantoms([alfa_v, alfa_h])  # BxTxH #####

        b0, b1, b2, b3 = tf.split(b_v, 4, axis=-1)
        f3 = tf.pow(f_odd, 3)
        f_sqrt = tf.sqrt(f_odd)
        b3 = tf.multiply(f_odd, b3)  # BxH #####
        b2 = tf.multiply(f3, b2)  # BxH #####
        b1 = tf.multiply(b1, f_sqrt)  # BxH #####
        decay_rates_odd = self.compute_decay_rate(b0, b1, b2, b3, t_odd)
        decay_rates_odd = tf.multiply(decay_rates_odd, m.pi)
        decay_odd = tf.math.exp(-decay_rates_odd)
        decay_odd = tf.multiply(decay_odd, alfas_odd[0])

        b0, b1, b2, b3 = tf.split(b_h, 4, axis=-1)
        f3 = tf.pow(f_odd, 3)
        f_sqrt = tf.sqrt(f_odd2)
        b3 = tf.multiply(f_odd, b3)  # BxH #####
        b2 = tf.multiply(f3, b2)  # BxH #####
        b1 = tf.multiply(b1, f_sqrt)  # BxH #####
        decay_rates_odd2 = self.compute_decay_rate(b0, b1, b2, b3, t_odd)
        decay_rates_odd2 = tf.multiply(decay_rates_odd2, m.pi)
        decay_odd2 = tf.math.exp(-decay_rates_odd2)
        decay_odd2 = tf.multiply(decay_odd2, alfas_odd[0])
        
        decay_odd3 = decay_odd2

        return decay_v, decay_h, attackTime_s, attackTime_t[:, :, 0], [decay_even, decay_even2, decay_odd, decay_odd2, decay_odd3]  # BxTxH
