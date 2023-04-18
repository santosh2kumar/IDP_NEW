from tensorflow.keras.models import load_model
import pickle

import pandas as pd
import numpy as np
import flask
import io

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from flask import request, jsonify

import warnings
warnings.filterwarnings('ignore')

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

cols_normalize = ['BPR', 'NRc', 'NRf', 'Nc', 'Nf', 'Nf_dmd', 'Ps30', 'T2', 'T24', 'T30',
       'T50', 'W31', 'W32', 'epr', 'farB', 'htBleed', 'setting_1',
       'setting_2']

def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

dependencies = {
    'r2_keras': r2_keras
}

model_lstm = load_model('model-00084-928.00182-0.77142-0.73563.h5', custom_objects=dependencies)

f = open('min_max_scaler.pkl','rb')
scaler = pickle.load(f)

@app.route("/api/v0/predict_rul", methods=["GET", "POST"])
def predict_rul():
    payload = request.stream
    dataset = pd.read_json(payload, typ='frame', orient='split')
    dataset.columns=['time_in_cycles', 'setting_1', 'setting_2', 'T2', 'T24', 'T30', 'T50', 'Nf', 'Nc', 'epr', 'Ps30', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'W31', 'W32']
    dataset[cols_normalize] = scaler.transform(dataset[cols_normalize])
    dataset = np.asarray(dataset).astype(np.float32)
    dataset = np.expand_dims(dataset, axis=0)
    RUL = model_lstm.predict(dataset)
    
    response = [{'RUL': int(RUL[0][0])}]
    return flask.jsonify(response)

if __name__ == "__main__":
	app.run(host='0.0.0.0')

