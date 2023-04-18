import pandas as pd
import numpy as np
import datetime
import os

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense , LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from pylab import rcParams
import math
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seq_array')
parser.add_argument('--label_array')
args = parser.parse_args()
seq_array = np.load(args.seq_array)
label_array = np.load(args.label_array)

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def lstm_train(seq_array, label_array, sequence_length):
    # The first layer is an LSTM layer with 128 units followed by another LSTM layer with 64 units. 
    # Dropout is also applied after each LSTM layer to control overfitting.
    # Then an additional Dense layer is applied with 32 neurons for better learning
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()
    model.add(LSTM(
             input_shape=(sequence_length, nb_features),
             units=256,
             return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(
              units=128,
              return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae',r2_keras])

    print(model.summary())
    curr_dt_time = datetime.datetime.now()
    # fit the network # Commoly used 100 epoches but 50-60 are fine its an early cutoff 
    model_name = '/mnt/model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
    if not os.path.exists(model_name):
        os.mkdir(model_name)
        
    filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{r2_keras:.5f}-{val_r2_keras:.5f}.h5'
    callbacks = [#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min'),
                 keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', period=1, verbose=1)]
    history = model.fit(seq_array, label_array, epochs=100, batch_size=256, validation_split=0.1, verbose=2, callbacks=callbacks)

    # list all data in history
    print(history.history.keys())
    
    return model, history

sequence_length = 100
model_instance, history = lstm_train(seq_array, label_array, sequence_length)

model_instance.save('/app/last_model.h5')
