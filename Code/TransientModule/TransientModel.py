import tensorflow as tf

def create_tmodel():
    freq_inputs = tf.keras.Input(shape=(1), name='freq_input')
    vel_inputs = tf.keras.Input(shape=(1), name='vel_inputs')
    act = 'tanh'
    inputs = tf.concat([freq_inputs, vel_inputs], axis=-1)

    out = tf.keras.layers.Dense(32)(inputs)
    out = tf.expand_dims(out, axis=-1)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(2, 2, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(4, 2, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(8, 2, activation=act, padding='valid')(out)
    out = tf.keras.layers.UpSampling1D(2)(out)
    out = tf.keras.layers.Conv1D(8, 4, activation=act, padding='valid')(out)

    out = tf.keras.layers.Conv1D(16, 4, activation=act, padding='valid')(out)

    out = tf.keras.layers.Conv1D(16, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)

    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 4, activation=act, padding='valid')(out)

    out = tf.keras.layers.Conv1D(32, 8, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 8, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 8, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 8, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 8, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 8, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 16, activation=act, padding='valid')(out)
    out = tf.keras.layers.Conv1D(32, 13, activation=act, padding='valid')(out)

    out = tf.keras.layers.Conv1D(1, 4, activation=act, padding='valid')(out)
    model = tf.keras.Model([freq_inputs, vel_inputs], out)
    model.summary()
    return model