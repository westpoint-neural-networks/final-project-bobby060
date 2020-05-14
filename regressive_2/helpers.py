import os
import scipy.io.wavfile as wav
import numpy as np
import tensorflow as tf
debug = False

def read_wav_as_np(file):
    # wav.read returns the sampling rate per second  (as an int) and the data (as a numpy array)
    rate, data = wav.read(file)
    # Normalize 16-bit input to [-1, 1] range
    np_arr = data.astype('float32') / 32767.0
    #np_arr = np.array(np_arr)
    return np_arr, data[0]


def saveAudio(arr, sample_rate, path):
    arr = arr * 32867.0
    arr = np.reshape(arr, (arr.shape[0],1))
    saved = tf.audio.encode_wav(arr ,sample_rate)
    tf.io.write_file(path, saved, name=None)
    return


# Essentially as is from unnati-xyz
def write_np_as_wav(X, sample_rate, file):
    # Converting the tensor back to it's original form
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    # wav.write constructs the .wav file using the specified sample_rate and tensor
    wav.write(file, sample_rate, Xnew)
    return

# Essentially as is from unnati-xyz
def convert_sample_blocks_to_np_audio(blocks):
    # Flattens the blocks into a single list
    song_np = np.concatenate(blocks)
    return song_np

# Essentially as is from unnati-xyz
def convert_np_audio_to_sample_blocks(song_np, block_size):

    # Block lists initialised
    block_lists = []

    # total_samples holds the size of the numpy array
    total_samples = song_np.shape[0]
    if debug:
        print('total_samples=',total_samples)

    # num_samples_so_far is used to loop through the numpy array
    num_samples_so_far = 0

    while (num_samples_so_far < total_samples):

        # Stores each block in the "block" variable
        block = song_np[num_samples_so_far:num_samples_so_far + block_size]

        if (block.shape[0] < block_size):
            # this is to add 0's in the last block if it not completely filled
            padding = np.zeros((block_size - block.shape[0],))
            # block_size is 44100 which is fixed throughout whereas block.shape[0] for the last block is <=44100
            block = np.concatenate((block,padding))
        block_lists.append(block)
        num_samples_so_far += block_size
    return block_lists

# Essentially as is from unnati-xyz
def time_blocks_to_fft_blocks(blocks_time_domain):
    # FFT blocks initialized
    fft_blocks = []
    for block in blocks_time_domain:
        # Computes the one-dimensional discrete Fourier Transform and returns the complex nD array
        # i.e The truncated or zero-padded input, transformed from time domain to frequency domain.
        fft_block = np.fft.fft(block)
        # Joins a sequence of blocks along frequency axis.
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(new_block)
    return fft_blocks

# Essentially as is from unnati-xyz
def fft_blocks_to_time_blocks(blocks_ft_domain):
    # Time blocks initialized
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        # Extracts real part of the amplitude corresponding to the frequency
        real_chunk = block[0:int(num_elems)]
        # Extracts imaginary part of the amplitude corresponding to the frequency
        imag_chunk = block[int(num_elems):]
        # Represents amplitude as a complex number corresponding to the frequency
        new_block = real_chunk + 1.0j * imag_chunk
        # Computes the one-dimensional discrete inverse Fourier Transform and returns the transformed
        # block from frequency domain to time domain
        time_block = np.fft.ifft(new_block)
        # Joins a sequence of blocks along time axis.
        time_blocks.append(time_block)
    return time_blocks



# Modified from unnati-xyz to allow for generating from multiple songs
def getSequences(path, bs, max_seq_len):

    wav_array, bitrate = read_wav_as_np(path)


# wav_array is converted into blocks with zeroes padded to fill the empty space in last block if any
# Zero padding makes computations easier and better for neural network
    wav_blocks_zero_padded = convert_np_audio_to_sample_blocks(wav_array, bs)
    if debug:
        print("len blocks 0 padded: ", len(wav_blocks_zero_padded))

# Flattens the blocks into an array
# Flattens the blocks into an array


# Shifts one left to create labels for training
    labels_wav_blocks_zero_padded = wav_blocks_zero_padded[1:]

    # Fast fourier transforming the wav blocks into frequency domain
    if debug:
        print('Dimension of wav blocks before fft: ',np.shape(wav_blocks_zero_padded))

    X = time_blocks_to_fft_blocks(wav_blocks_zero_padded)
    Y = time_blocks_to_fft_blocks(labels_wav_blocks_zero_padded)
    print('num fft blocks: ',len(X))
    if debug:
        print('Dimension of the training dataset (wav blocks after fft): ',np.shape(X))

    cur_seq = 0
    chunks_X = []
    chunks_Y = []
    total_seq = len(X)
    while cur_seq + max_seq_len < total_seq:
        chunks_X.append(X[cur_seq:cur_seq + max_seq_len])
        chunks_Y.append(Y[cur_seq:cur_seq + max_seq_len])
        cur_seq += max_seq_len
    # Number of examples
    num_examples = len(chunks_X)
    # Imaginary part requires the extra space
    num_dims_out = bs * 2

    # Originally returned np arrays, returns list because I np later
    return chunks_X, chunks_Y


# Modified from unnati-xyz to allow for generating from multiple songs
def getSequences2(path, bs, max_seq_len):

    wav_array, bitrate = read_wav_as_np(path)


# wav_array is converted into blocks with zeroes padded to fill the empty space in last block if any
# Zero padding makes computations easier and better for neural network
    wav_blocks_zero_padded = convert_np_audio_to_sample_blocks(wav_array, bs)
    if debug:
        print("len blocks 0 padded: ", len(wav_blocks_zero_padded))

# Flattens the blocks into an array
# Flattens the blocks into an array


# Shifts one left to create labels for training
    labels_wav_blocks_zero_padded = wav_blocks_zero_padded[1:]

    # Fast fourier transforming the wav blocks into frequency domain
    if debug:
        print('Dimension of wav blocks before fft: ',np.shape(wav_blocks_zero_padded))

    X = time_blocks_to_fft_blocks(wav_blocks_zero_padded)
    Y = time_blocks_to_fft_blocks(labels_wav_blocks_zero_padded)
    print('num fft blocks: ',len(X))
    if debug:
        print('Dimension of the training dataset (wav blocks after fft): ',np.shape(X))

    cur_seq = 0
    chunks_X = []
    chunks_Y = []
    total_seq = len(X)
    while cur_seq + max_seq_len < total_seq:
        chunks_X.append(X[cur_seq:cur_seq + max_seq_len])
        # Only change to v2, creates a single target instead of a sequence.
        chunks_Y.append(Y[cur_seq + max_seq_len-1])
        cur_seq += max_seq_len
    # Number of examples
    num_examples = len(chunks_X)
    # Imaginary part requires the extra space
    num_dims_out = bs * 2

    # Originally returned np arrays, returns list because I np later
    return chunks_X, chunks_Y
