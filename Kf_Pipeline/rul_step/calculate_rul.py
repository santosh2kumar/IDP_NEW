import pandas as pd
import numpy as np

fd_001_train = pd.read_csv("/mnt/fd_train.csv", header=None)
fd_001_test = pd.read_csv("/mnt/fd_test.csv", header=None)

fd_001_train.drop(columns=[26,27],inplace=True)
fd_001_test.drop(columns=[26,27],inplace=True)

columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

fd_001_train.columns = columns
fd_001_test.columns = columns

fd_001_train.drop(columns=['PCNfR_dmd','TRA'],inplace=True)
fd_001_test.drop(columns=['PCNfR_dmd','TRA'],inplace=True)

def prepare_train_data(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]

df = prepare_train_data(fd_001_train)
unit_number = pd.DataFrame(df["unit_number"])
train_df = df.drop(columns = ['unit_number','P15','P2','P30','phi'])
fd_001_test = fd_001_test.drop(columns = ['P15','P2','P30','phi'])
RUL = pd.read_csv("/mnt/fd_RUL.csv", header=None)

unit_number.to_csv('/app/unit_number.csv', index=False)
train_df.to_csv('/app/train_df.csv', index=False)
fd_001_test.to_csv('/app/fd_001_test.csv', index=False)
RUL.to_csv('/app/RUL.csv', index=False)

print(unit_number.head())
print(train_df.head())
print(fd_001_test.head())
print(RUL.head())
