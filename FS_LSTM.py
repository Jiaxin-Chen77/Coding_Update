#!/opt/anaconda2/envs/python3/bin/python

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import MaxPooling1D
# from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Convolution1D, Dropout
# from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras import regularizers
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Reshape
# from tensorflow.keras.layers import Attention
# from tensorflow.keras.utils import plot_model

# import math
import time
import numpy as np
# import pandas as pd
import random
# from scipy.io import loadmat
# import datetime
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import pickle
def save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def cal_RMSE(origin, pred):
    mse = np.average((pred-origin)**2)
    rmse = np.sqrt(mse)

    return rmse


# ################ Model Variables ################
target_feature = 0 # target feature order in table [0,2,3,4,7,9,10,11,14,16,17,18]
n_past = 24
n_future = 12

# ################ Prepare Dataset ###################
table = load('Fill_Gap/Imputed_Measurement')
table.columns = ['1Wave Height (m)', '1Max Wave Height (m)', '1Tpeak (s)', '1Tz (s)',
       '1Peak Direction (degrees)', '1Spread (degrees)', '1Sea Temp (C)',
       '2Wave Height (m)', '2Max Wave Height (m)', '2Tpeak (s)', '2Tz (s)',
       '2Peak Direction (degrees)', '2Spread (degrees)', '2Sea Temp (C)',
       '3Wave Height (m)', '3Max Wave Height (m)', '3Tpeak (s)', '3Tz (s)',
       '3Peak Direction (degrees)', '3Spread (degrees)', '3Sea Temp (C)']

train_df, test_df = table[:6000], table[6000:7200]

# Scaling table
train = train_df.copy()
scalers={}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s
test = test_df.copy()
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1, 1))
    s_s=np.reshape(s_s, len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s


# ################# Feature Selection #####################
def feature_select_input():
    table_scale = table.copy()
    scalers_FS={}
    for i in table.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        s_s = scaler.fit_transform(table_scale[i].values.reshape(-1, 1))
        s_s=np.reshape(s_s, len(s_s))
        scalers_FS['scaler_'+ i] = scaler
        table_scale[i]=s_s
    # To ensure the accuracy of the n_future, I/O set as current/n_future
    X_FS, y_FS = table_scale.values[:-1*n_future], \
                 table_scale.iloc[n_future:, target_feature].values  # Scalar table
    return X_FS, y_FS


def feature_selection(X, y):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.feature_selection import SelectFromModel
    ENCV = ElasticNetCV(cv=5, random_state=0,tol=0.0001, selection='random').fit(X, y)
    model = SelectFromModel(ENCV, prefit=True)
    X_embed = model.transform(X)
    list_column_number = []
    list_column = []
    rn = random.randint(1,len(table))
    for i in X_embed[rn]:
        for j in range(len(X[rn])):
            if i == X[rn][j]:
                list_column_number.append(j)
                list_column.append(table.columns[j])
    return list_column_number, list_column


# FS_order = feature_selection(feature_select_input()[0], feature_select_input()[1])[0]


# Generate Training Dataset
def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
    # slicing the past and future parts of the window
    #     if series.shape == 2:
        past, future = series[window_start:past_end], series[past_end:future_end]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


# Generate Model
def build_model_lstm(train, n_past, n_future, FS_order):
    # prepare data
    train_x, train_y = split_series(train.values, n_past, n_future)
    train_x = train_x[:, :, FS_order]
    train_y = train_y[:, :, target_feature]
    # train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    # train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)
    #     train_y = train_y[:,:,0]
    # define parameters
    verbose, epochs, batch_size = 0, 25, 16
    n_features_input = train_x.shape[2]
    n_features_output = 1 # Single output
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features_input)))
    model.add(RepeatVector(n_future))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(n_features_output)))

    #     reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.09 ** x)
    #     model.compile(loss='mean_absolute_percentage_error', optimizer='adam') # learning_rate=1e-3, decay_rate=0.9
    model.compile(loss='mse', optimizer='adam')  # learning_rate=1e-3, decay_rate=0.9
    #     # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, validation_split=0.2)
    return model, history


# train the model
def build_model_StackedLSTM(train, n_past, n_future, FS_order):
    # prepare data
    train_x, train_y = split_series(train.values, n_past, n_future)
    train_x = train_x[:, :, FS_order]
    train_y = train_y[:, :, [target_feature]]
    # define parameters
    verbose, epochs, batch_size = 0, 25, 16
    n_features_input = train_x.shape[2]
    n_features_output = 1
    #     n_features = train_x.shape[2]
    # define model
    encoder_inputs = Input(shape=(n_past, n_features_input))
    encoder_l1 = LSTM(200, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    #
    decoder_inputs = RepeatVector(n_future)(encoder_outputs1[0])
    #
    decoder_l1 = LSTM(200, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l1 = TimeDistributed(tf.keras.layers.Dense(100))(decoder_l1)
    decoder_outputs1 = TimeDistributed(tf.keras.layers.Dense(n_features_output))(decoder_l1)
    #
    model = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.09 ** x)
    model.compile(loss='mse', optimizer='adam')  # learning_rate=1e-3, decay_rate=0.9
    #     # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              verbose=verbose, callbacks=[reduce_lr], validation_split=0.2)

    return model, history


# train the model
def build_model_2StackedLSTM(train, n_past, n_future, FS_order):
    # prepare data
    train_x, train_y = split_series(train.values, n_past, n_future)
    # define parameters
    train_x = train_x[:, :, FS_order]
    train_y = train_y[:, :, [target_feature]]
    # define parameters
    verbose, epochs, batch_size = 0, 25, 16
    n_features_input = train_x.shape[2]
    n_features_output = 1
    # define model
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features_input))
    encoder_l1 = tf.keras.layers.LSTM(200, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(200, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    #
    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
    #
    decoder_l1 = tf.keras.layers.LSTM(200, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(200, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_l2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100))(decoder_l2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features_output))(decoder_l2)
    #
    model = tf.keras.models.Model(encoder_inputs, decoder_outputs2)

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.09 ** x)
    model.compile(loss='mse', optimizer='adam')  # learning_rate=1e-3, decay_rate=0.9
    #     # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, callbacks=[reduce_lr], validation_split=0.2)

    return model, history


def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        scores.append(rmse)
            # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


FS_order = feature_selection(feature_select_input()[0], feature_select_input()[1])[0]
# X_train, y_train = split_series(train.values, n_past, n_future)


# ##################### Train Model ############################
t0 = time.time()
# model, history = build_model_lstm(train, n_past, n_future, FS_order)
# model, history = build_model_StackedLSTM(train, n_past, n_future, FS_order)
model, history = build_model_2StackedLSTM(train, n_past, n_future, FS_order)

print('Train_Val_loss',np.mean(history.history['loss'][10:]), np.mean(history.history['val_loss'][10:]))

# ##################### Predict and Compare ##############################
X_test, y_test = split_series(test.values, n_past, n_future)
prd_LSTM_FS_1buoy = model.predict(X_test[:, :, FS_order])
prd_LSTM_FS_1buoy[:, :, 0]=scalers['scaler_'+table.columns[target_feature]].inverse_transform(prd_LSTM_FS_1buoy[:, :, 0])
y_test[:, :, target_feature] = scalers['scaler_' +
                                         table.columns[target_feature]].inverse_transform(y_test[:, :, target_feature])

print(evaluate_forecasts(y_test[:, :, target_feature], prd_LSTM_FS_1buoy[:, :, 0]))

print('Time Consuming: ', time.time()-t0)

