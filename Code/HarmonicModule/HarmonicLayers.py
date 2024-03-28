import tensorflow as tf
import math as m
import numpy as np

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

        f_n = tf.divide(freq_inputs, self.max_)#BxD

        B = self.linear(f_n) #BxU
        B = self.linearOut(B) #Bx1

        #####law3
        n2 = tf.pow(self.n, 2) #BxH
        Bn = tf.keras.layers.Multiply()([n2, B]) #BxH
        Bn = tf.keras.layers.Add()([self.ones, Bn]) #BxH

        final_n = tf.keras.layers.Multiply()([self.n, Bn]) #BxH
        freqs_inputs = tf.keras.layers.Multiply()([final_n, freq_inputs])  # BxDxH

        return freqs_inputs, f_n, B, final_n


class DeltaLayer(tf.keras.layers.Layer):
    def __init__(self, B=1, Fs=24000, trainable=True, type=tf.float32):
        super(DeltaLayer, self).__init__()
        self.type = type
        self.max_ = tf.constant(494.62890625, dtype=self.type)
        self.twopi = tf.constant(2 * m.pi, dtype=self.type)
        self.Fs = Fs
        self.B = B
        self.trainable = trainable

        self.linear_t2 = tf.keras.layers.Dense(128, name='t2', input_shape=(self.B, 2), trainable=self.trainable, dtype=self.type)
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
        t = tf.multiply(t, ti) # BxTxH

        k = tf.expand_dims(index_inputs, axis=-1)  # Bx1x1
        k = tf.multiply(k, self.num_steps)
        t = t + k
        x = tf.expand_dims(x, axis=1)  # Bx1xH
        partials = tf.multiply(x, t, name='mul2')  # BxTxH
        return partials, f, t


class DecayLayer(tf.keras.layers.Layer):
    def __init__(self, B, harmonics=24, Fs=24000, trainable=True, double_pol=False, type=tf.float32):
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
       
        self.alfaIn = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8, name='alfaIn', batch_input_shape=(self.B, 3), trainable=self.trainable, dtype=self.type))
        self.alfa = tf.keras.layers.LSTM(16, stateful=True, return_sequences=False, name='alfa', trainable=self.trainable, dtype=self.type)
        self.alfaOut = tf.keras.layers.Dense(self.harmonics*self.p, name='alfaOut', activation=None, trainable=self.trainable, dtype=self.type)
        self.alfaNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.attackTime1 = tf.keras.layers.Dense(8, name='attackTime1', trainable=self.trainable, dtype=self.type)
        self.attackTime2 = tf.keras.layers.Dense(1, name='attackTime2', trainable=self.trainable, dtype=self.type)
      
        self.bIn = tf.keras.layers.Dense(8, name='bIn', batch_input_shape=(self.B, 3), trainable=self.trainable, dtype=self.type)
        self.b = tf.keras.layers.Dense(16, name='b', trainable=self.trainable, dtype=self.type)
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

    def __call__(self, f_n, vel_inputs, f, t, K):
        all_inp = tf.keras.layers.Concatenate(axis=-1)([f_n, vel_inputs])  # BxDx2

        attackTime_s = self.attackTime1(all_inp)  # Bx2
        attackTime_s = self.attackTime2(attackTime_s)  # Bx1
        attackTime_s = tf.nn.sigmoid(attackTime_s)  # Bx1
        attackTime_s = tf.expand_dims(attackTime_s, axis=1)  # Bx1x1
        attackTime_t = tf.multiply(attackTime_s, t)# BxTx1
        attackTime_t = tf.math.minimum(attackTime_t, 1.)

        all_inp = tf.keras.layers.Concatenate(axis=-1)([all_inp, K])  # Bx3
        all_inp = tf.expand_dims(all_inp, axis=-1)

        f3 = tf.pow(f, 3)
        t_s = tf.divide(t, self.Fs)
        f_sqrt = tf.sqrt(f)

        alfa_v = self.alfaIn(all_inp)  # Bx1xH
        alfa_v = self.alfa(alfa_v)  # BxDxH
        alfa_v = self.alfa2(alfa_v)  # BxDxH
        alfa_v = self.alfaNorm(alfa_v)  # BxDxH
        alfa_v = self.alfaOut(alfa_v) # BxDxH

        b_v = self.bIn(all_inp[:,:,0])  # BxH
        b_v = self.b(b_v)  # BxH
        b_v = self.bNorm(b_v)  # BxH
        b_v = self.bOut(b_v) # BxH
        b_v = tf.nn.relu(b_v)
        
        alfa_v = tf.nn.sigmoid(alfa_v)  # BxDxH
        alfa_v = tf.divide(alfa_v, self.harmonics*2)

        alfa_v, alfa_h = tf.split(alfa_v, [self.harmonics, self.harmonics], axis=-1)
        
        b_v, b_h = tf.split(b_v, [4, 4], axis=-1)

        b4, b5, b6, b7 = tf.split(b_h, 4, axis=-1)

        b7 = tf.multiply(f, b7)  # BxDxH #####
        b6 = tf.multiply(f3, b6)  # BxDxH #####
        b5 = tf.multiply(b5, f_sqrt)
        
        alfa_h = tf.expand_dims(alfa_h, axis=1)  # Bx1xH
        decay_rates_h = self.compute_decay_rate(b4, b5, b6, b7, t_s)
        decay_rates_h = tf.multiply(decay_rates_h, m.pi)
        decay_h = tf.math.exp(-decay_rates_h)
        decay_h = tf.multiply(decay_h, alfa_h)

        
        b0, b1, b2, b3 = tf.split(b_v, 4, axis=-1)

        b3 = tf.multiply(f, b3)  # BxH #####
        b2 = tf.multiply(f3, b2)  # BxH #####
        b1 = tf.multiply(b1, f_sqrt) # BxH #####
        
        alfa_v = tf.expand_dims(alfa_v, axis=1)  # Bx1xH
        decay_rates_v = self.compute_decay_rate(b0, b1, b2, b3, t_s) # BxTxH #####
        decay_rates_v = tf.multiply(decay_rates_v, m.pi)
        decay_v = tf.math.exp(-decay_rates_v)
        decay_v = tf.multiply(decay_v, alfa_v)
        decay_v = tf.multiply(attackTime_t, decay_v)

        decay_h = tf.multiply(attackTime_t, decay_h)

        return decay_v, decay_h, attackTime_s  # BxTxH