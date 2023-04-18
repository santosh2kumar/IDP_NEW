import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_df')
parser.add_argument('--fd_001_test')
args = parser.parse_args()
train_df = pd.read_csv(args.train_df)
fd_001_test = pd.read_csv(args.fd_001_test)


X = train_df.iloc[:,:19].to_numpy() 
Y = train_df.iloc[:,19:].to_numpy()
Y = np.ravel(Y)


test_max = fd_001_test.groupby('unit_number')['time_in_cycles'].max().reset_index()
test_max.columns = ['unit_number','max']
fd_001_test = fd_001_test.merge(test_max, on=['unit_number'], how='left')
test = fd_001_test[fd_001_test['time_in_cycles'] == fd_001_test['max']].reset_index()
test.drop(columns=['unit_number','index','max'],inplace = True)

X_001_test = test.to_numpy()

np.save('/app/X.npy', X)
np.save('/app/Y.npy', Y)
np.save('/app/X_001_test.npy', X_001_test)

print(X[0:5])
print(Y[0:5])
print(X_001_test[0:5])
