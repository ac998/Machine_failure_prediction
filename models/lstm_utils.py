import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.layers import Dense , LSTM, Dropout
from keras.models import Sequential, load_model
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import rcParams
import math

def prepare_train_dataset(data):

    df = data.copy()
    # drop columns op_set_3, sm_1, sm_5, sm_10, sm_16, sm_18, sm_19 as they have constant values (std = 0)
    # drop op_set_1, op_set_2 because they have low correlation with the output. 
    # drop sm_14 because it's highly correlated with sm_9
    df.drop(columns=['op_set_3', 'sm_1', 'sm_5', 'sm_10', 'sm_16', 'sm_18', 'sm_19','sm_14', 'op_set_1', 'op_set_2'], 
                  inplace=True)
    rul = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul = pd.DataFrame(rul)
    rul.columns = ['unit_number', 'last_cycle']
    df = df.merge(rul, on=['unit_number'], how='left')
    df['rul'] = df['last_cycle'] - df['time_in_cycles']
    df.drop(columns=['last_cycle'], inplace=True)
    return df[df['time_in_cycles'] > 0]


def prepare_test_dataset(data):
    df = data.copy()
    # drop features not used in train set
    df.drop(columns=['op_set_3', 'sm_1', 'sm_5', 'sm_10', 'sm_16', 'sm_18', 'sm_19', 'sm_14', 'op_set_1', 'op_set_2'], 
                 inplace=True)
    return df

def gen_sequence(id_df, seq_length, seq_cols):
    """ 
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones 
    """
    # for one id, put all the rows in a single matrix
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

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # Remove the first seq_length labels because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]


def lstm_preprocessing(raw_train_df, raw_test_df, raw_truth_df):
    train_df = raw_train_df
    test_df = raw_test_df
    truth_df = raw_truth_df
    
    # Normalize columns except [id , cycle, rul]
    cols_normalize = train_df.columns.difference(['unit_number','time_in_cycles', 'rul']) 
    # MinMax normalization (from 0 to 1)
    min_max_scaler = MinMaxScaler()

    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=train_df.index)
    # Train set
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns = train_df.columns)
    
    # Test set
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test_df.index)
    
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)
    
    # We use the ground truth dataset to generate labels for the test data.
    # generate column max for test data
    rul = pd.DataFrame(test_df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number','max']
    truth_df.columns = ['more']
    truth_df['unit_number'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more'] # adding true-rul vlaue + max cycle of test data set w.r.t M_ID
    truth_df.drop('more', axis=1, inplace=True)

    # generate RUL for test data
    test_df = test_df.merge(truth_df, on=['unit_number'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['time_in_cycles']
    test_df.drop('max', axis=1, inplace=True) 

    ## pick a large window size of 50 cycles
    sequence_length = 50
    
     # pick the feature columns 
    sequence_cols = list(test_df.columns[:-3])
    
    # generator for the sequences transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], sequence_length, sequence_cols)) 
               for id in train_df['unit_number'].unique())

    # convert generated sequences to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    # generate labels
    label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['rul'])  for id in train_df['unit_number'].unique()]

    label_array = np.concatenate(label_gen).astype(np.float32)
    
    return seq_array, label_array, test_df, sequence_length, sequence_cols

def R2(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def lstm_model(seq_array, label_array, sequence_length):
    # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
    # Dropout is also applied after each LSTM layer to control overfitting. 
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()
    model.add(LSTM(input_shape=(sequence_length, nb_features), units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', R2])

    print(model.summary())
    return model


def test_model(lstm_test_df, model, sequence_length, sequence_cols):
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
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
    print('MAE: {}'.format(scores_test[1]))
    print('\nR^2: {}'.format(scores_test[2]))

    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last

    test_set = pd.DataFrame(y_pred_test)

    # Plot in blue color the predicted data and in orange color the actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_test)
    plt.plot(y_true_test, color="orange")
    plt.title('prediction')
    plt.ylabel('RUL')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()

    return scores_test[1], scores_test[2]