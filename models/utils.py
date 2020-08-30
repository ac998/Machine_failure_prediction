import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

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
    df.drop(columns=['last_cycle', 'unit_number'], inplace=True)
    return df[df['time_in_cycles'] > 0]



def prepare_test_dataset(data):
    df = data.copy()
    # drop features not used in train set
    df.drop(columns=['op_set_3', 'sm_1', 'sm_5', 'sm_10', 'sm_16', 'sm_18', 'sm_19', 'sm_14', 'op_set_1', 'op_set_2'], 
                 inplace=True)
    
    last_cycle = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    last_cycle.columns = ['unit_number','max']
    df = df.merge(last_cycle, on=['unit_number'], how='left')
    test_df = df[df['time_in_cycles'] == df['max']].reset_index()
    test_df.drop(columns=['index', 'max', 'unit_number'], inplace=True)
    return test_df


def plot_signals(raw_train_df, features, engine_number):
    fig, axes = plt.subplots(ncols=3, nrows=5, constrained_layout=True, figsize=(15,15))
    axes = axes.ravel()
    for i, feature in enumerate(features):
        column = raw_train_df.groupby('unit_number')[feature]
        data = column.get_group(engine_number)
        axes[i].plot(data)
        axes[i].set_title(feature)
        
    plt.show()

def score(y_test, y_pred):
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    print("Mean absolute error (in cycles) : {}".format(mae))
    print("Root mean square error (in cycles) : {}".format(rmse))
    print("Coefficent of determination : {}".format(r2))
    return
