from __future__ import absolute_import, division, print_function, unicode_literals, generators

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
keras.__version__
from tensorflow.keras import layers

generator_input = keras.Input(shape=(sample_length,1))
# https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
x = layers.LSTM(100, return_sequences=True)(generator_input)
x = layers.LSTM(250, return_sequences=True)(x)
x = layers.LSTM(100, return_sequences=True)(x)
x = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)

generator = keras.models.Model(generator_input,x)
generator.summary()

discriminator_input = layers.Input(shape=(sample_length, 1))

x = layers.LSTM(250)(discriminator_input)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = keras.Input(shape=(sample_length, 1))
gan_output = discriminator(generator(gan_input))

gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0,decay=1e-8)
gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')


