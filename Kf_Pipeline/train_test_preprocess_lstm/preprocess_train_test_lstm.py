import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_df')
parser.add_argument('--fd_001_test')
parser.add_argument('--unit_number')
parser.add_argument('--RUL')

args = parser.parse_args()
unit_number = pd.read_csv(args.unit_number)
train_df = pd.read_csv(args.train_df)
fd_001_test = pd.read_csv(args.fd_001_test)
RUL = pd.read_csv(args.RUL)


def lstm_data_preprocessing(raw_train_data, raw_test_data, raw_RUL_data):
    train_df = raw_train_data
    test_df = raw_test_data
    truth_df = raw_RUL_data
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    
    #################
    # TRAIN
    #################
    
    # MinMax normalization (from 0 to 1)
    cols_normalize = train_df.columns.difference(['unit_number','time_in_cycles','RUL']) # NORMALIZE COLUMNS except [id , cycle, rul ....]

    min_max_scaler = MinMaxScaler()

    train_df[cols_normalize] = min_max_scaler.fit_transform(train_df[cols_normalize])

    print("train_df >> ",train_df.head())
    print("\n")

    
    #################
    # TEST
    #################
    
    # MinMax normalization (from 0 to 1)
    test_df[cols_normalize] = min_max_scaler.transform(test_df[cols_normalize])
    
    
    # generate RUL for test data
    rul = pd.DataFrame(test_df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number','max']
    truth_df.columns = ['more']
    truth_df['unit_number'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more'] # adding true-rul vlaue + max cycle of test data set
    truth_df.drop('more', axis=1, inplace=True)

    
    test_df = test_df.merge(truth_df, on=['unit_number'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['time_in_cycles']
    test_df.drop('max', axis=1, inplace=True) 

    
    # pick a large window size of 50 cycles
    sequence_length = 100

    # function to reshape features into (samples, time steps, features) 
    def gen_sequence(id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,112),(50,192)
        # 0 50 -> from row 0 to row 50
        # 1 51 -> from row 1 to row 51
        # 2 52 -> from row 2 to row 52
        # ...
        # 111 191 -> from row 111 to 191
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    # pick the feature columns 
    sequence_cols = list(train_df.columns[1:-1])

    print(sequence_cols)

    # generator for the sequences
    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], sequence_length, sequence_cols)) 
               for id in train_df['unit_number'].unique())

    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    print(seq_array.shape)

    # function to generate labels
    def gen_labels(id_df, seq_length, label):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        # For one id I put all the labels in a single matrix.
        # For example:
        # [[1]
        # [4]
        # [1]
        # [5]
        # [9]
        # ...
        # [200]] 
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        # I have to remove the first seq_length labels
        # because for one id the first sequence of seq_length size have as target
        # the last label (the previus ones are discarded).
        # All the next id's sequences will have associated step by step one label as target.
        return data_matrix[seq_length:num_elements, :]

    # generate labels
    label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['RUL']) 
                 for id in train_df['unit_number'].unique()]

    label_array = np.concatenate(label_gen).astype(np.float32)
    print(label_array.shape)
    print(label_array)
    
    return seq_array, label_array, test_df, sequence_length, sequence_cols

train_df_lstm = pd.concat([unit_number, train_df], axis=1)
seq_array, label_array, test_df, sequence_length, sequence_cols = lstm_data_preprocessing(train_df_lstm, fd_001_test.copy(), RUL.copy())

np.save('/app/seq_array.npy', seq_array)
np.save('/app/label_array.npy', label_array)
test_df.to_csv('/app/test_df.csv', index=False)

print(seq_array[0:5])
print(label_array[0:5])
print(test_df.head())

