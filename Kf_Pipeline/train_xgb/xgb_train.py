import pandas as pd
import numpy as np
import pickle
import xgboost

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--X')
parser.add_argument('--Y')

args = parser.parse_args()
X = np.load(args.X)
Y = np.load(args.Y)

model_xgb = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.018, gamma=0, subsample=0.8,
                           colsample_bytree=0.5, max_depth=5,silent=True)
model_xgb.fit(X,Y)

with open('/app/model_xgb.pickle', 'wb') as f:
    pickle.dump(model_xgb,f)
