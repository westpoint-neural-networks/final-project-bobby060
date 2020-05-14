import os
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

import pickle
import sys
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Dense, LSTM
from helpers import getSequences,fft_blocks_to_time_blocks, saveAudio, convert_sample_blocks_to_np_audio

# Original code copied from here https://github.com/unnati-xyz/music-generation/blob/master/MusicGen.ipynb

# tf.logging.set_verbosity(tf.logging.ERROR)
debug = False
# define block size

"""
Usage:

train_model.py <mode> <modelpath> <epochs> <songpath>

Three modes:
"new" creates new model at path
"continue" loads old model, trains
"generate" only generates a song

"""
argv = sys.argv
mode = 0
epochs = 0
model_path = argv[2]
song_path = argv[4]
if argv[1]=="generate":
    mode = 2
elif argv[1]=="continue":
    mode = 1
    epochs = int(argv[3])
else:
    epochs = int(argv[3])

strategy = tf.distribute.OneDeviceStrategy (device="/GPU:3")
num_gpus = strategy.num_replicas_in_sync
with strategy.scope():
    sample_frequency = 16000'
    trainpath = '../../train/'
    testpath = '../../test/'
    max_seq_len = 10
    # bs = 44100
    bs = 16000

    # Will contain list of song blocks.

    x_test = []
    y_test = []

    def train_generator(batch = 20):
        while True:
            for file in os.listdir(trainpath):
                # Decodes audio
                if file.endswith(".wav"):
                    path = trainpath+file
                    x, y = getSequences(path, bs, max_seq_len)
                    if debug:
                        print(" Number sequences generated: ",len(x))
                    out_shape_train = (len(x), max_seq_len, bs * 2)
                    x_train_arr = np.zeros(out_shape_train)
                    y_train_arr = np.zeros(out_shape_train)

                    for n in range(len(x)):
                        for i in range(max_seq_len):
                            x_train_arr[n][i] = x[n][i]
                            y_train_arr[n][i] = y[n][i]

                    yield x_train_arr, y_train_arr

    def test_generator(batch = 20):
        while True:
            for file in os.listdir(testpath):
                if debug:
                    print('walking test')
                # Decodes audio
                if file.endswith(".wav"):
                    path = testpath + file
                    x,y = getSequences(path, bs, max_seq_len)
                    if debug:
                        print(" Number sequences generated: ",len(x))
                    out_shape_test = (len(x), max_seq_len, bs * 2)
                    x_test_arr = np.zeros(out_shape_test)
                    y_test_arr = np.zeros(out_shape_test)

                    for n in range(len(x)):
                        for i in range(max_seq_len):
                            x_test_arr[n][i] = x[n][i]
                            y_test_arr[n][i] = y[n][i]

                    yield x_test_arr, y_test_arr

    num_frequency_dimensions = 32000
    num_hidden_dimensions = 1024
    print('Input layer size: ',num_frequency_dimensions)
    print('Hidden layer size: ',num_hidden_dimensions)
    # Sequential is a linear stack of layers

    if mode == 0:
        model = Sequential()
        # This layer converts frequency space to hidden space
        model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(10, bs*2)))
        # return_sequences=True implies return the entire output sequence & not just the last output
        model.add(LSTM(num_hidden_dimensions, return_sequences=True))
        # This layer converts hidden space back to frequency space
        model.add(TimeDistributed(Dense(bs*2)))
        # Done building the model.Now, configure it for the learning process
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mean_squared_error'])

        model.summary()

    else:
        model = load_model(model_path)

    batch = 20

    test_gen = test_generator(20)
    train_gen = train_generator(20)

    if mode !=2:
    # Number of iterations for training
        num_iters = epochs
        # Number of iterations before we save our model
        epochs_per_iter = 10
        # Number of training examples pushed to the GPU per batch.
        batch_size = 5
        # Path to weights file
        cur_iter = 0
        while cur_iter < num_iters:
            print('Iteration: ' + str(cur_iter))
            # Iterate over the training data in batches
            history = model.fit_generator(train_gen,  steps_per_epoch= 500, epochs=epochs_per_iter, validation_data=test_gen, validation_steps=50)
            model.save(model_path)
            with open('regressionhistory5', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            cur_iter += epochs_per_iter
        print('Training complete!')

    # We take the first chunk of the training data itself for seed sequence.
    seed_seq = next(train_gen)[0][0]
    # Reshaping the sequence to feed to the RNN.
    seed_seq = np.reshape(seed_seq, (1, seed_seq.shape[0], seed_seq.shape[1]))

    # Generated song sequence is stored in output.
    output = []
    for it in range(max_seq_len):
        # Generates new value
        seedSeqNew = model.predict(seed_seq)
        # Appends it to the output
        if it == 0:
            for i in range(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy())
        # newSeq contains the generated sequence.
        next_step = seedSeqNew[0][-1]
        next_step = np.reshape(next_step, (1, next_step.shape[0]))
        newSeq = np.concatenate((seed_seq[0][-9:], next_step), axis=0)
        if debug:
            print('next step shape: ', newSeq.shape)
        # Reshaping the new sequence for concatenation.
        newSeq = np.reshape(newSeq, (1, newSeq.shape[0], newSeq.shape[1]))
        # Appending the new sequence to the old sequence.
        seed_seq = newSeq


    # The path for the generated song
    # Reversing the conversions
    time_blocks = fft_blocks_to_time_blocks(output)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    saveAudio(song, sample_frequency, song_path)
