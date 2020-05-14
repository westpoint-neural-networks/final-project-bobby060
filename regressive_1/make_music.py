from __future__ import absolute_import, division, print_function, unicode_literals, generators

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
keras.__version__
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from tensorflow.keras import layers
import ffmpeg
import IPython.display as ipd

from pydub import AudioSegment
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, History

from keras.layers import Dense, LSTM, LeakyReLU, Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential, load_model

from sklearn.preprocessing import normalize, MinMaxScaler

import sys

"""

Use to create songs from an existing model using the step by step framework
Usage: python make_music <model_path> <input_window> <bitrate> <num songs> <savepath> 

"""
argv = sys.argv
bitrate = int(argv[3])
model_path = argv[1]
num_songs = int(argv[4])
savepath = argv[5]
input_window=int(argv[2])


def saveAudio(arr, path):
    saved = tf.audio.encode_wav(tf.cast(arr, float) ,22000)
    tf.io.write_file(path, saved, name=None)

sample_counter = 1

def get_start_sample():
	data_location = "../data/test"

	for subdir, dirs, files in os.walk(data_location):
          for file in files:
              #print os.path.join(subdir, file)
              filepath = subdir + os.sep + file
              # Decodes audio
              if filepath.endswith(".mp3"):
                    mp3_audio = AudioSegment.from_file(filepath, format="mp3")
                    # rudimentary downsample factor of 3
                    audio_array = mp3_audio.get_array_of_samples()
                    audio_array = np.array(audio_array)
                    audio_array = audio_array.astype('float32')/30000
                    audio_array = np.nan_to_num(audio_array, nan=0.0)
                    samplename = "sample"+str(sample_counter)+"seed.wav"
                    path = model_path+"/"+samplename
                    mp3_audio.export(path, format="wav")
                    return audio_array[1000:1000+input_window]

def song_generator(lookback,model, starter,len = 20000):
    newsong = np.zeros(len+lookback)
    for i in range(lookback):
        newsong[i] = starter[i]
    print("copied song...")
    for i in range(len-lookback-1):
        newsong[lookback+i] = model.predict(newsong[lookback+i:i+2*lookback].reshape(1,lookback,1))
        if i%1000==0:
            print("Predicted " + str(i)+ " samples")
    scaler = make_reverse_transform()
    output = newsong.reshape(len,1)
    output = scaler.inverse_transform(output)
    output = output.reshape(1,len)
    return newsong[lookback:]
    
model = load_model(model_path)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

strategy = tf.distribute.OneDeviceStrategy (device="/GPU:3")

with strategy.scope():
	for i in range(num_songs):
		start_sample = get_start_sample()
		newsong = song_generator(input_window, model, start_sample)
		export_path = savepath+"/sample_generated_" + str(sample_counter)+".wav"
		saveAudio(newsong.reshape(20000,1)*30000, export_path)
		sample_counter+=1




