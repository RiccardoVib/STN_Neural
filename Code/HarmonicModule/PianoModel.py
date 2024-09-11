import tensorflow as tf
from Layers import InharmonicLayer, DecayLayer, SinuosoidsGenerator, DeltaLayer
import matplotlib.pyplot as plt
import math as m
from LayersNoise import NoiseGenerator


class PianoModel(tf.keras.Model):
    def __init__(self, batch_size=1, num_steps=240, num_frames=240, harmonics=24, fs=24000, max_steps=2799.0, g=9,
                 phan=False, train_b=True, train_phase=False, train_amps=False, train_S=False, type=tf.float32):
        super(PianoModel, self).__init__()

        self.twopi = tf.constant(2 * m.pi, dtype=type)
        self.g = g
        self.batch_size = batch_size
        self.harmonics = harmonics
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.num_frames = num_frames
        self.train_S = train_S
        self.train_phase = train_phase
        self.train_b = train_b
        self.train_amps = train_amps
        self.h_even = self.harmonics - self.g
        self.h_odd = self.h_even * 2 - 2
        self.phan = phan

        self.InharmonicModel = InharmonicLayer(harmonics=harmonics, batch_size=self.batch_size, num_frames=num_frames,
                                               Fs=fs, trainable=self.train_b, type=type)
        self.SinuosoidsGenerator = SinuosoidsGenerator(batch_size=self.batch_size, harmonics=harmonics,
                                                       num_steps=num_steps, num_frames=self.num_frames, type=type)
        self.SinuosoidsGenerator_odd = SinuosoidsGenerator(batch_size=self.batch_size, harmonics=self.h_odd,
                                                           num_steps=num_steps, num_frames=self.num_frames, type=type)
        self.SinuosoidsGenerator_even = SinuosoidsGenerator(batch_size=self.batch_size, harmonics=self.h_even,
                                                            num_steps=num_steps, num_frames=self.num_frames, type=type)
        self.DecayModel = DecayLayer(batch_size=self.batch_size, harmonics=harmonics, trainable=self.train_amps,
                                     num_frames=self.num_frames, g=self.g, phan=self.phan, type=type)
        self.deltas = DeltaLayer(trainable=self.train_amps, num_frames=self.num_frames, type=type)

    def __call__(self, inputs, training=False):
        freq_inputs = tf.reshape(inputs[0],
                                 [self.batch_size // self.num_frames, self.num_frames, 1])  # BxDx1  #[self.B, 1]
        index_inputs = tf.reshape(inputs[1],
                                  [self.batch_size // self.num_frames, self.num_frames, 1])  # BxDx1  #[self.B, 1]
        vel_inputs = tf.reshape(inputs[2],
                                [self.batch_size // self.num_frames, self.num_frames, 1])  # BxDx1  #[self.B, 1]
        B_inputs = tf.reshape(inputs[3],
                              [self.batch_size // self.num_frames, self.num_frames, 1])  # BxDx1  #[self.B, 1]
        attackTime = tf.reshape(inputs[4],
                                [self.batch_size // self.num_frames, self.num_frames, 1])  # BxDx1  #[self.B, 1]

        ######Compute Frequencies
        ##### first polarization
        partials_predicted, final_n_c = self.InharmonicModel(freq_inputs, B_inputs)
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted, 11000),
                                       tf.zeros_like(partials_predicted), 1)
        partials_predicted_loss = partials_predicted
        partials_predicted = amplitude_envelopes * partials_predicted

        ##### second polarization +/- 0.1/0.2 hz
        delta_T2 = self.deltas(vel_inputs)
        partials_h = tf.add(delta_T2, freq_inputs)
        partials_h = tf.keras.layers.Multiply()([final_n_c, partials_h])  # BxDxH
        amplitude_envelopes = tf.where(tf.greater_equal(partials_h, 11000), tf.zeros_like(partials_h), 1)
        partials_h = amplitude_envelopes * partials_h

        if self.phan:
            #### Odd
            partials_predicted_odd = []
            for i in range(self.g, partials_predicted.shape[-1] - 1):
                partials_predicted_odd.append(tf.add(partials_predicted[:, :, i], partials_predicted[:, :, i + 1]))
                partials_predicted_odd.append(
                    tf.abs(tf.add(partials_predicted[:, :, i], -partials_predicted[:, :, i + 1])))
            partials_predicted_odd = tf.transpose(tf.convert_to_tensor(partials_predicted_odd), [1, 2, 0])
            amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_odd, 11000),
                                           tf.zeros_like(partials_predicted_odd), 1)
            partials_predicted_odd = amplitude_envelopes * partials_predicted_odd

            #### even
            partials_predicted_even = tf.multiply(partials_predicted[:, :, self.g:], 2.)
            amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_even, 11000),
                                           tf.zeros_like(partials_predicted_even), 1)
            partials_predicted_even = amplitude_envelopes * partials_predicted_even

            #### Odd
            partials_predicted_odd2 = []
            for i in range(self.g, partials_h.shape[-1] - 1):
                partials_predicted_odd2.append(tf.add(partials_h[:, :, i], partials_h[:, :, i + 1]))
                partials_predicted_odd2.append(tf.abs(tf.add(partials_h[:, :, i], -partials_h[:, :, i + 1])))
            partials_predicted_odd2 = tf.transpose(tf.convert_to_tensor(partials_predicted_odd2), [1, 2, 0])
            amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_odd2, 11000),
                                           tf.zeros_like(partials_predicted_odd2), 1)
            partials_predicted_odd2 = amplitude_envelopes * partials_predicted_odd2

            #### even2
            partials_predicted_even2 = tf.multiply(partials_h[:, :, self.g:], 2.)
            amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_even2, 11000),
                                           tf.zeros_like(partials_predicted_even2), 1)
            partials_predicted_even2 = amplitude_envelopes * partials_predicted_even2

            #### Odd mixed
            partials_predicted_odd3 = []
            for i in range(self.g, partials_h.shape[-1] - 1):
                partials_predicted_odd3.append(tf.add(partials_predicted[:, :, i], partials_h[:, :, i + 1]))
                partials_predicted_odd3.append(tf.abs(tf.add(partials_predicted[:, :, i], -partials_h[:, :, i + 1])))
            partials_predicted_odd3 = tf.transpose(tf.convert_to_tensor(partials_predicted_odd3), [1, 2, 0])
            #### even
            amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_odd3, 11000),
                                           tf.zeros_like(partials_predicted_odd3), 1)
            partials_predicted_odd3 = amplitude_envelopes * partials_predicted_odd3

        #####Compute Sines

        partials_xt, f, t = self.SinuosoidsGenerator(partials_predicted, index_inputs)
        partials_xt_h, f_h, _ = self.SinuosoidsGenerator(partials_h, index_inputs)

        if self.phan:
            partials_xt_even, f_even, t_even = self.SinuosoidsGenerator_even(partials_predicted_even, index_inputs)
            partials_xt_even2, f_even2, _ = self.SinuosoidsGenerator_even(partials_predicted_even2, index_inputs)
            partials_xt_odd, f_odd, t_odd = self.SinuosoidsGenerator_odd(partials_predicted_odd, index_inputs)
            partials_xt_odd2, f_odd2, _ = self.SinuosoidsGenerator_odd(partials_predicted_odd2, index_inputs)
            partials_xt_odd3, f_odd3, _ = self.SinuosoidsGenerator_odd(partials_predicted_odd3, index_inputs)

            partials = [f_even, f_even2, f_odd, f_odd2, f_odd3]

            decay_v, decay_h, attackTime_t, decays_long = self.DecayModel(vel_inputs, [f, f_h], t,
                                                                          index_inputs / self.max_steps,
                                                                          partials, t_even, t_odd, attackTime)
        else:

            decay_v, decay_h, attackTime_t = self.DecayModel(vel_inputs, [f, f_h], t, index_inputs / self.max_steps,
                                                             f, t, t, attackTime)
        #####phase
        # phase = self.phase
        # phase = self.phaseNorm(phase)
        # phase = tf.nn.sigmoid(phase)
        # phase = tf.multiply(phase, self.twopi)
        # phase = tf.expand_dims(phase, axis=0)
        # phase = tf.repeat(phase, self.harmonics, axis=1)
        # phase = tf.expand_dims(phase, axis=0)
        # phase = tf.expand_dims(phase, axis=0)
        # phase = tf.repeat(phase, self.batch_size // self.num_frames, axis=0)
        #
        # phase_v, phase_h = tf.split(phase, [self.harmonics, self.harmonics], axis=-1)
        ####
        #
        # all_harmonics = tf.sin(partials_xt+phase_v)
        all_harmonics = tf.sin(partials_xt)
        #decay_v = tf.multiply(decay_v, tf.expand_dims(attackTime_t, axis=-1))
        all_harmonics = tf.multiply(decay_v, all_harmonics)
        all_harmonics = tf.math.reduce_sum(all_harmonics, axis=-1, name='reduce_harmonics')  # sum harmonics

        # all_harmonics_h = tf.sin(partials_xt_h+phase_h)
        all_harmonics_h = tf.sin(partials_xt_h)
        #decay_h = tf.multiply(decay_h, tf.expand_dims(attackTime_t, axis=-1))
        all_harmonics_h = tf.multiply(decay_h, all_harmonics_h)
        all_harmonics_h = tf.math.reduce_sum(all_harmonics_h, axis=-1, name='reduce_harmonics')  # sum harmonics
        all_harmonics = tf.add(all_harmonics_h, all_harmonics)

        if self.phan:
            ## long
            all_harmonics_even = tf.sin(partials_xt_even)
            decays_long0 = decays_long[0]
            #decays_long0 = tf.multiply(decays_long[0], tf.expand_dims(attackTime_t, axis=-1))
            all_harmonics_even = tf.multiply(decays_long0, all_harmonics_even)
            all_harmonics_even = tf.math.reduce_sum(all_harmonics_even, axis=-1, name='reduce_harmonics')  # sum harmonics
            all_harmonics = tf.add(all_harmonics_even, all_harmonics)
            
            all_harmonics_even = tf.sin(partials_xt_even2)
            decays_long1 = decays_long[1]
            #decays_long1 = tf.multiply(decays_long[1], tf.expand_dims(attackTime_t, axis=-1))
            all_harmonics_even = tf.multiply(decays_long1, all_harmonics_even)
            all_harmonics_even = tf.math.reduce_sum(all_harmonics_even, axis=-1, name='reduce_harmonics')  # sum harmonics
            all_harmonics = tf.add(all_harmonics_even, all_harmonics)

            all_harmonics_odd = tf.sin(partials_xt_odd)
            decays_long2 = decays_long[2]
            #decays_long2 = tf.multiply(decays_long[2], tf.expand_dims(attackTime_t, axis=-1))
            all_harmonics_odd = tf.multiply(decays_long2, all_harmonics_odd)
            all_harmonics_odd = tf.math.reduce_sum(all_harmonics_odd, axis=-1, name='reduce_harmonics')  # sum harmonics
            all_harmonics = tf.add(all_harmonics_odd, all_harmonics)

            all_harmonics_odd = tf.sin(partials_xt_odd2)
            decays_long3 = decays_long[3]
            #decays_long3 = tf.multiply(decays_long[3], tf.expand_dims(attackTime_t, axis=-1))
            all_harmonics_odd = tf.multiply(decays_long3, all_harmonics_odd)
            all_harmonics_odd = tf.math.reduce_sum(all_harmonics_odd, axis=-1, name='reduce_harmonics')  # sum harmonics
            all_harmonics = tf.add(all_harmonics_odd, all_harmonics)

            all_harmonics_odd = tf.sin(partials_xt_odd3)
            decays_long4 = decays_long[4]
            #decays_long4 = tf.multiply(decays_long[4], tf.expand_dims(attackTime_t, axis=-1))
            all_harmonics_odd = tf.multiply(decays_long4, all_harmonics_odd)
            all_harmonics_odd = tf.math.reduce_sum(all_harmonics_odd, axis=-1, name='reduce_harmonics')  # sum harmonics
            all_harmonics = tf.add(all_harmonics_odd, all_harmonics)


        alfa = tf.math.reduce_max(tf.abs(all_harmonics), axis=-1, keepdims=True)
        rms = tf.abs(tf.reduce_mean(tf.square(all_harmonics), axis=-1, keepdims=True))

        #decay_v = decay_v[:, :, 0, :6]

        if self.train_b:
            return [partials_predicted_loss[:, :, :6]]
        elif self.train_amps:
            return [all_harmonics, rms, alfa]
        elif self.train_S:
            return [all_harmonics]
        else:
            return [partials_predicted_loss[:, :, :6], all_harmonics, rms, alfa]
