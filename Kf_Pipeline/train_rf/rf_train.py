import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--X')
parser.add_argument('--Y')

args = parser.parse_args()
X = np.load(args.X)
Y = np.load(args.Y)

model_rf = RandomForestRegressor(n_estimators=70, max_features=7, max_depth=5, n_jobs=-1, random_state=1)
model_rf.fit(X,Y)

with open('/app/model_rf.pickle', 'wb') as f:
    pickle.dump(model_rf,f)
