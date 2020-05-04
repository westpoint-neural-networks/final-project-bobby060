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


def saveAudio(arr, path):
    saved = tf.audio.encode_wav(tf.cast(arr, float) ,22000)
    tf.io.write_file(path, saved, name=None)


def train_sequence_generator(lookback = 25, bs = 200):
    data_location = "../data/train"
    data_array = np.zeros((bs,lookback))
    label_array = np.zeros((bs, 1))
    counter = 0
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
                    audio_array = audio_array.astype('float32')
                    audio_array = audio_array/30000
                    for i in range (0,len(audio_array) - lookback - 1,100):
                        data = audio_array[i:i+lookback]
#                         data = data.reshape((1,lookback,1))
                        label = audio_array[i+lookback+1:i+lookback+2]
                        label.reshape((1,1))
                        data_array[counter] = data
                        label_array[counter] = label
                        counter +=1
                        if(counter == bs):
                            counter = 0
                            out_data = data_array.reshape(bs,lookback,1)
                            out_labels = label_array.reshape(bs,1)
                            yield (out_data,out_labels)
                        
def test_sequence_generator(lookback = 25, bs = 200):
    data_location = "../data/test"
    data_array = np.zeros((bs,lookback))
    label_array = np.zeros((bs, 1))
    counter = 0
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
                    audio_array = audio_array.astype('float32')
                    audio_array = audio_array/30000
                    for i in range (0,len(audio_array) - lookback - 1,100):
                        data = audio_array[i:i+lookback]
#                         data = data.reshape((1,lookback,1))
                        label = audio_array[i+lookback+1:i+lookback+2]
                        label.reshape((1,1))
                        data_array[counter] = data
                        label_array[counter] = label
                        counter +=1
                        if(counter == bs):
                            counter = 0
                            out_data = data_array.reshape(bs,lookback,1)
                            out_labels = label_array.reshape(bs,1)
                            yield (out_data,out_labels)
                        
def song_generator(lookback,model, starter,len = 20000):
    newsong = np.zeros(len+lookback)
    for i in range(lookback):
        newsong[i] = starter[i]
    print("copied song...")
    for i in range(len-lookback-1):
        newsong[lookback+i] = model.predict(newsong[lookback+i:i+2*lookback].reshape(1,lookback,1))
        if i%1000==0:
            print("Predicted " + str(i)+ " samples")
    return newsong[lookback:]
    

fp = 'models/bestrnnmodel.h5'
mc_cp = ModelCheckpoint(filepath = fp, save_best_only = True, verbose = 1)
    
es_cb = EarlyStopping(monitor = 'val_loss', mode='min', verbose = 1, patience = 5, min_delta=0.0001, restore_best_weights=True)
cb_list = [mc_cp]

strategy = tf.distribute.OneDeviceStrategy (device="/GPU:3")
num_gpus = strategy.num_replicas_in_sync
with strategy.scope():
	regression_model = Sequential()
	regression_model.add(LSTM(100, activation='linear', input_shape=(None, 1)))
	regression_model.add(LeakyReLU())
	regression_model.add(Dense(50, activation='linear'))
	regression_model.add(LeakyReLU())
	regression_model.add(Dense(25, activation='linear'))
	regression_model.add(LeakyReLU())
	regression_model.add(Dense(12, activation='linear'))
	regression_model.add(LeakyReLU())
	regression_model.add(Dense(units=1, activation='linear'))
	regression_model.add(LeakyReLU())


	regression_model.compile(optimizer='adam', loss='mean_squared_error')
	regression_model.summary()


	regression_model2 = Sequential()
	regression_model2.add(Conv1D(32, 5, activation='linear', input_shape=(None, 1)))
	regression_model2.add(LeakyReLU())
	regression_model2.add(MaxPooling1D(3))
	regression_model2.add(Conv1D(32, 5, activation='linear'))
	regression_model2.add(LeakyReLU())
	regression_model2.add(LSTM(500, activation='linear', recurrent_dropout=0.3))
	regression_model2.add(LeakyReLU())
	regression_model2.add(Dense(250, activation='linear'))
	regression_model2.add(Dropout(0.5))
	regression_model2.add(LeakyReLU())
	regression_model2.add(Dense(25, activation='linear'))
	regression_model2.add(LeakyReLU())
	regression_model2.add(Dense(12, activation='linear'))
	regression_model2.add(LeakyReLU())
	regression_model2.add(Dense(units=1, activation='linear'))
	regression_model2.add(LeakyReLU())


	regression_model2.compile(optimizer='adam', loss='mean_squared_error')
	regression_model2.summary()
	print(tf.config.experimental.list_physical_devices('GPU'))


	lb = 200
	batchsize = 500

	train_gen = train_sequence_generator(lookback = lb, bs = batchsize)
	test_gen = test_sequence_generator(lookback = lb, bs = batchsize)


	history = regression_model.fit_generator(train_gen, 
	                                         steps_per_epoch = 100,
	                                         epochs = 5,
	                                         validation_data=test_gen,
	                                         validation_steps = 5,
	                                         callbacks = cb_list)

	# https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
	# with open('/regression2history', 'wb') as file_pi:
	#         pickle.dump(history.history, file_pi)

	regression_model.save('models/regression_model3.hd5')
	        
	gendata, res = next(test_gen)

	newsong = song_generator(200, regression_model, gendata[20])
	saveAudio(newsong.reshape(20000,1)*30000, 'results/regressionmodel1output.wav')


