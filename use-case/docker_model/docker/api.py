import flask
from flask import request, jsonify
import numpy as np
import json
import sys
import pandas as pd
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = True
def pred_vect(v):
    vect = {}
    j = 1
    for i in range (0,28):
        ind = "V"+str(j)
        vect [ind] = v[i]
        j = j+1
    vect ["Amount"] = v[28]
    return pd.DataFrame([vect])

filename = 'model.pkl'

#post a json object to see the prediction for it :
@app.route('/foo', methods=['POST']) 
def foo():
    print (request)
    #return request
    return json.dumps(request.json)

@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    content = request.get_json()
    print (content)
    return jsonify(content)

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Fraud Detection API : IOO</h1>
<p> A POC for the use of Machine learning to detect Paysim Mobile Transaction Frauds.</p>'''

@app.route('/api/v0/verify', methods=['GET','POST'])
def predict_new_transaction():
    inp = request.json
    vec = [[inp['amount'],inp['oldBalanceOrig'],inp['newBalanceOrig'],inp['oldBalanceDest'],inp['newBalanceDest'],inp['step_day'],inp['hour'],inp['step_week'],inp['CASH_OUT'],inp['DEBIT'],inp['PAYMENT'],inp['TRANSFER'],inp['CM'],inp['errorOrig'],inp['errorDest']]]
    f = open(filename,'rb')
    loaded_model = pickle.load(f)
    f.close()
    res = str(loaded_model.predict(vec)[0])
    result = [
    {'id': 0,
     'prediction': res} ]
    return jsonify(result)

@app.route('/api/v0/test', methods=['GET','POST'])
def predict_test():
    f = open(filename,'rb')
    loaded_model = pickle.load(f)
    f.close()
    res = str(loaded_model.predict(pred_vect(v))[0])
    result = [
    {'id': 0,
     'prediction': res} ]
    return jsonify(result)

@app.route('/api/v0/info', methods=['GET'])
def info():
    result = [
        {
            'Author' : 'IOO',
            'description' : 'A fraud detection model using a kaggle dataset',
        }
    ]
    return jsonify(result)  
app.run(host='0.0.0.0')
