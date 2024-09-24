"""
This code is customized from https://github.com/magenta/ddsp
"""
import tensorflow as tf
import numpy as np
from scipy import fftpack
from Utils import tf_float32

def apply_window_to_impulse_response(impulse_response, window_size):

    impulse_response = tf_float32(impulse_response)

    # If IR is in causal form, put it in zero-phase form.
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.shape[-1]) #2*(noise_bands-1)
    if (window_size > ir_size):
        window_size = ir_size
    window = tf.signal.hann_window(window_size)

    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = tf.concat([window[half_idx:],
                          tf.zeros([padding]),
                          window[:half_idx]], axis=0)
    else:
        window = tf.signal.fftshift(window, axes=-1)

    # Apply the window, to get new IR (both in zero-phase form).
    window = tf.broadcast_to(window, impulse_response.shape)
    impulse_response = window * tf.math.real(impulse_response)

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = tf.concat([impulse_response[..., first_half_start:],
                                    impulse_response[..., :second_half_end]],
                                   axis=-1)
    else:
        impulse_response = tf.signal.fftshift(impulse_response, axes=-1)
    return impulse_response


def fft_convolve(audio, impulse_response, padding='same', delay_compensation=-1):
    audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

    # Get shapes of audio.
    batch_size, frames, audio_size = audio.shape.as_list()

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape.as_list()
    if len(ir_shape) == 2:
      impulse_response = impulse_response[:, tf.newaxis, :]

    # Broadcast impulse response.
    if ir_shape[0] == 1 and batch_size > 1:
        impulse_response = tf.tile(impulse_response, [batch_size, 1, 1])

    # Get shapes of impulse response.
    ir_shape = impulse_response.shape.as_list()
    batch_size_ir, n_ir_frames, ir_size = ir_shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                       'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=True)

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = tf.signal.rfft(audio_frames, [fft_size])
    ir_fft = tf.signal.rfft(impulse_response, [fft_size])

    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = tf.multiply(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_out = tf.signal.irfft(audio_ir_fft)
    #audio_out = tf.signal.overlap_and_add(audio_out, hop_size)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(audio_out[0,:,:,:], audio_size, ir_size, padding, delay_compensation)


# Time-varying convolution -----------------------------------------------------
def get_fft_size(frame_size, ir_size, power_of_2):

    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size


def crop_and_compensate_delay(audio, audio_size, ir_size, padding='same', delay_compensation=-1):
    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError('Padding must be \'valid\' or \'same\', instead of {}.'.format(padding))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return audio[:, :, start:-end]

########
def frequency_impulse_response(magnitudes, window_size):

    # Get the IR (zero-phase form). Cast tensor as if it was complex adding a zeroed imaginary part.
    magnitudes = tf.complex(magnitudes, tf.zeros_like(magnitudes))
    # Compute the impulse response of the filters.
    # This gives a zero-centered response.
    impulse_response = tf.signal.irfft(magnitudes) #impulse = 2*(magnitudes-1)

    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response,
                                                        window_size)

    return impulse_response


class NoiseGenerator(tf.keras.layers.Layer):
    def __init__(self, window_size, ir_size, trainable=False, max_steps=2799.0, type=tf.float32):#ir_size => window_size
        """
        Noise generator object
            :param window_size: input size
            :param ir_size: size of the impuslse response to compute
            :param trainable: if train the layers
        """
        super(NoiseGenerator, self).__init__()

        # Amplitude envelope as complex OLA
        self.window_size = window_size
        self.ir_size = ir_size
        self.hop_size = self.window_size // 2
        self.padding = self.window_size // 2
        self.trainable = trainable
        self.type = type
        self.max_steps = max_steps

        self.coeff = tf.keras.layers.Dense(self.window_size, activation=None, name='impulse', trainable=self.trainable, dtype=self.type)####relu
        self.amps = tf.keras.layers.Dense(1, name='amps', activation='sigmoid', trainable=self.trainable, dtype=self.type)
        self.mean = tf.keras.layers.Dense(1, name='mean', activation='tanh', trainable=self.trainable, dtype=self.type)

    def __call__(self, vel_inputs, K):

        K = tf.divide(K, self.max_steps)
        all_inp = tf.keras.layers.Concatenate(axis=-1)([vel_inputs, K])

        noise_bands = self.coeff(all_inp)###fft_length = 2 * (inner - 1) se noise_bands=128 -> impulse 127*2

        amps = self.amps(all_inp)
        mean = self.mean(all_inp)
    
        # Create a sequence of IRs according to input.
        impulse = frequency_impulse_response(noise_bands, self.ir_size) #self.ir_size <= (noise_bands-1)*2 = (self.window_size-1)*2

        # compute noise and filter it via convolution
        noise = tf.random.normal([impulse.shape[0],  self.window_size], mean=mean, stddev=1.0, seed=422, dtype=self.type, name='noise')###mean
        noise = noise/tf.math.reduce_max(noise)

        impulse = tf.expand_dims(impulse, axis=1)
        noise = tf.expand_dims(noise, axis=1)

        noise = fft_convolve(noise, impulse, 'same', -1)
        noise = tf.signal.overlap_and_add(noise, self.window_size//2)

        noise = tf.multiply(amps, noise)

        return noise, mean


