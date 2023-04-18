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
from tensorflow.keras.models import load_model
from pylab import rcParams
import math
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_lstm')
parser.add_argument('--test_df')
args = parser.parse_args()


test_df = pd.read_csv(args.test_df)


def score(y_true,y_pred,a1=10,a2=13):
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0 :
            score += math.exp(i/a2) - 1   
        else:
            score += math.exp(- i/a1) - 1
    return score

def score_func(y_true,y_pred):
    lst = [round(score(y_true,y_pred),2), 
          round(mean_absolute_error(y_true,y_pred),2),
          round(mean_squared_error(y_true,y_pred),2)**0.5,
          round(r2_score(y_true,y_pred),2)]
    
    print(f' compatitive score {lst[0]}')
    print(f' mean absolute error {lst[1]}')
    print(f' root mean squared error {lst[2]}')
    print(f' R2 score {lst[3]}')
    return [lst[1], round(lst[2],2), lst[3]*100]

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def lstm_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols):
    # We pick the last sequence for each id in the test data
    seq_array_test_last = [lstm_test_df[lstm_test_df['unit_number']==id][sequence_cols].values[-sequence_length:] 
                           for id in lstm_test_df['unit_number'].unique() if len(lstm_test_df[lstm_test_df['unit_number']==id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Similarly, we pick the labels
    y_mask = [len(lstm_test_df[lstm_test_df['unit_number']==id]) >= sequence_length for id in lstm_test_df['unit_number'].unique()]
    label_array_test_last = lstm_test_df.groupby('unit_number')['RUL'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

    estimator = model

    # test metrics
    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last
    score_eval = score_func(y_true_test, y_pred_test)
    test_set = pd.DataFrame(y_pred_test)
    #print(test_set.head())
    return score_eval[0], score_eval[1], score_eval[2]

dependencies = {
    'r2_keras': r2_keras
}
model_lstm = load_model(args.model_lstm, custom_objects=dependencies)
sequence_length=100
sequence_cols=['time_in_cycles', 'setting_1', 'setting_2', 'T2', 'T24', 'T30', 'T50', 'Nf', 'Nc', 'epr', 'Ps30', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'W31', 'W32']
model_metrics = lstm_valid_evaluation(test_df, model_lstm, sequence_length, sequence_cols)


