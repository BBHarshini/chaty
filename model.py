import tensorflow as tf


def create_model(input_size, output_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_size, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size))
    return model
