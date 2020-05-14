import os
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

import pickle
import sys
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Dense, LSTM
from helpers import getSequences2,fft_blocks_to_time_blocks, saveAudio, convert_sample_blocks_to_np_audio

# Original code copied from here https://github.com/unnati-xyz/music-generation/blob/master/MusicGen.ipynb

# tf.logging.set_verbosity(tf.logging.ERROR)
debug = True
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
    sample_frequency = 44100
    trainpath = '../../musicrganold/yoyoma_dataset/train/'
    testpath = '../../musicrganold/yoyoma_dataset/test/'

    max_seq_len = 10
    bs = 44100

    # Will contain list of song blocks.
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file in os.listdir(trainpath):
        # Decodes audio
        if file.endswith(".wav"):
            path = trainpath+file
            x, y = getSequences2(path, bs, max_seq_len)
            if debug:
                print(" Number sequences generated: ",len(x))
            x_train.append(x)
            y_train.append(y)

    for file in os.listdir(testpath):
        if debug:
            print('walking test')
        # Decodes audio
        if file.endswith(".wav"):
            path = testpath + file
            x,y = getSequences2(path, bs, max_seq_len)
            if debug:
                print(" Number sequences generated: ",len(x))
            x_test.append(x)
            y_test.append(y)

    if debug:
        print(len(x_train), ' train songs read')
        print(len(x_test), ' test songs read')

    total_len_train = 0
    total_len_test = 0
    for x in x_train:
        total_len_train+=len(x)
    for x in x_test:
        total_len_test+=len(x)

    if debug:
        print(len(x_train[0]))
        print(' num train seqs created: ', total_len_train)
        print(' num test seqs createdL: ', total_len_test)

    out_shape_train = (total_len_train, max_seq_len, bs*2)
    label_shape_train = (total_len_train, bs*2)
    out_shape_test = (total_len_test, max_seq_len,bs*2)
    label_shape_test = (total_len_test, bs*2)
    x_train_arr = np.zeros(out_shape_train)
    y_train_arr = np.zeros(label_shape_train)

    x_test_arr = np.zeros(out_shape_test)
    y_test_arr = np.zeros(label_shape_test)

    offset = 0
    for x in range(len(x_train)):
        for n in range (len(x_train[x])):
            for i in range(max_seq_len):
                x_train_arr[n+offset][i] = x_train[x][n][i]
            y_train_arr[n + offset][0] = y_train[x][n][i]
        offset+=len(x_train[x])

    offset = 0
    for x in range(len(x_test)):
        for n in range(len(x_test[x])):
            for i in range(max_seq_len):
                x_test_arr[n + offset][i] = x_test[x][n][i]
            y_test_arr[n + offset][0] = y_test[x][n][0]
        offset += len(x_test[x])

    if debug:
        print(len(x_train_arr), ' train samples loaded')
        print(len(x_test_arr), 'test samples loaded')

    print(x_train_arr.shape)
    num_frequency_dimensions = x_train_arr.shape[1]
    num_hidden_dimensions = 1024
    print('Input layer size: ',num_frequency_dimensions)
    print('Hidden layer size: ',num_hidden_dimensions)
    # Sequential is a linear stack of layers

    if mode == 0:
        model = Sequential()
        # This layer converts frequency space to hidden space
        model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(num_frequency_dimensions, bs*2)))
        # return_sequences=True implies return the entire output sequence & not just the last output
        model.add(LSTM(num_hidden_dimensions))
        # This layer converts hidden space back to frequency space
        model.add(Dense(bs*2))
        # Done building the model.Now, configure it for the learning process
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mean_squared_error'])

        model.summary()

    else:
        model = load_model(model_path)

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
            history = model.fit(x_train_arr, y_train_arr, batch_size=batch_size, epochs=epochs_per_iter, validation_data=(x_test_arr, y_test_arr))
            model.save(model_path)
            with open('regressionhistory2', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            cur_iter += epochs_per_iter
        print('Training complete!')

    # We take the first chunk of the training data itself for seed sequence.
    seed_seq = x_train_arr[2]
    # Reshaping the sequence to feed to the RNN.
    seed_seq = np.reshape(seed_seq, (1, seed_seq.shape[0], seed_seq.shape[1]))

    # Generated song sequence is stored in output.
    output = []
    for it in range(max_seq_len):
        # Generates new value
        predicted= model.predict(seed_seq)
        # Appends it to the output
        output.append(predicted[0])
        # newSeq contains the generated sequence.
        next_step = predicted
        seed_seq = np.concatenate((seed_seq[0][-9:], predicted), axis=0)
        seed_seq = np.reshape(seed_seq, (1, seed_seq.shape[0], seed_seq.shape[1]))
        if debug:
            print("step done")


    # The path for the generated song
    # Reversing the conversions
    time_blocks = fft_blocks_to_time_blocks(output)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    saveAudio(song, sample_frequency, song_path)
