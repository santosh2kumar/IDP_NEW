import pickle
import math
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_rf')
parser.add_argument('--X_001_test')

args = parser.parse_args()

rf_f = open(args.model_rf,'rb')
model_rf = pickle.load(rf_f)
X_001_test = np.load(args.X_001_test)
RUL = pd.read_csv("/mnt/fd_RUL.csv", header=None)

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

y_pred_rf = model_rf.predict(X_001_test)
y_true = RUL[0].to_numpy()
RF_individual_scorelst = score_func(y_true, y_pred_rf)
print(RF_individual_scorelst)
