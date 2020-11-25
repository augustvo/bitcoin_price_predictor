# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for bitcoin_price_predictor Project
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
import random
#pd.random.seed(123)
#import seaborn as sns
import datetime
#from datetime import date
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bitcoin_price_predictor.params import START_DATE, INTERVAL, STOCKS, KEY_Values


def download_ydata():
    """method to get stock data from yahoo finance"""

    end = datetime.date.today()
    start = datetime.datetime(2019,11,1)
    print("Downloading data: ")
    sotck_prices = yf.download(STOCKS,start=start,end = end, interval=INTERVAL)

    a =[(KEY_Values,STOCKS) for key in KEY_Values for index in STOCKS]

    return sotck_prices[a]


def preproc_ydata():
    """method gets yahoo data ready to run the model"""

    data = download_ydata()

    data['date'] = data.index
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    data = data.fillna(method='ffill')

    data['percentage_change_price'] = data['Close']['BTC-USD'].pct_change()

    new_colum_names = []

    for el1, el2 in data.columns:
        new_colum_names.append(el1 + "_" + el2)

    data.columns = new_colum_names

    data['delta_binary'] = data.percentage_change_price_>0
    data['delta_binary'] = data['delta_binary'].astype(int)

    return data


def data_scaling():

  df = preproc_ydata()

  # Splitting 80/20
  start = 1 # do a for loop
  index = round((len(df)-start)*.8)

  df = df.drop(columns=['Date'])
  # Past
  df_train = df.iloc[start:index+start]
  #Future
  df_test = df.iloc[start+index:]

  temporal_horizon = 1 # -> predict the day after
  # min_len = 7
  # max_len = 8
  length_of_sequences = 7

  normalizer_features = MinMaxScaler()
  normalizer_features.fit(df_train)
  scaled_df = pd.DataFrame(normalizer_features.transform(df))
  scaled_df['delta_binary'] = df.reset_index()[['delta_binary']]

  scaled_df_train, scaled_df_test = scaled_df.iloc[start:(index+start)], scaled_df.iloc[start+index:]

  X_train, X_test, y_train, y_test, y_b_test, y_b_train = generate_data(scaled_df_train,
                                                    scaled_df_test, temporal_horizon, length_of_sequences)

  return X_train, X_test, y_train, y_test, y_b_test, y_b_train


def get_sample(length, temporal_horizon, random_start):
    """Method gets subsamples from intier time-series, each corresponding to
    one sequence of data Xi with its corresponding prediction yi;
    length -> correspondes to the lenght of the observed sequence;
    temporal_horizon -> corresponds to the number of days between last seen stock market
                        value and the day to predict"""

    data = preproc_ydata(data)

    features = [col_name for col_name in data.columns if col_name not in {'Date', 'Close_BTC-USD','delta_binary',
                                                                        'percentage_change_price_'} ]

    temporal_horizon = temporal_horizon - 1
    last_possible = data.shape[0] - temporal_horizon - length

    X_sample = data[features].iloc[random_start: random_start+length].values ## we need to say which features we are using??!
    y_sample = data['delta_binary'].iloc[random_start+length+temporal_horizon]
    y_before = data['delta_binary'].iloc[random_start+length+temporal_horizon-1]

    return X_sample, y_sample, y_before


def get_X_y(data, temporal_horizon, length_of_sequences, start_of_sequences):
    """Method runs the function get_sample several times to samples from the entire dataset.
    data -> corresponds to your input data
    temporal_horizon -> is the number of days between the last seen value and
                        the predicted one
    length_of_sequences -> is a list that corresponds to the length of each
                            sample 𝑋𝑖 : [len(X_1), len(X_2), len(X_3), ..., ] as
                            that each sequence 𝑋𝑖 has no reason to be of the same
                            length as the other one."""

    X, y, y_before = [], [], []

    for start in start_of_sequences:
        xi, yi, y_bi = get_sample(data, length_of_sequences, temporal_horizon, start)
        X.append(xi)
        y.append(yi)
        y_before.append(y_bi)

    return X, np.array(y), np.array(y_before)


def generate_data(df_train, df_test, temporal_horizon, length_of_sequences):
    """Method generates train and test sets"""

    np.random.seed(0)

    # Train
    last_possible = df_train.shape[0] - temporal_horizon - length_of_sequences
    random_start = np.random.randint(0, last_possible)

    start_of_sequences_train = np.random.randint(0, last_possible, df_train.shape[0])

    X_train, y_train, y_b_train = get_X_y(df_train, temporal_horizon, length_of_sequences, start_of_sequences_train)
    X_train = pad_sequences(X_train, padding='post', dtype='float32')

    # Test
    last_possible = df_test.shape[0] - temporal_horizon - length_of_sequences
    random_start = np.random.randint(0, last_possible)

    start_of_sequences_test = np.random.randint(0, last_possible, df_test.shape[0])

    X_test, y_test, y_b_test = get_X_y(df_test, temporal_horizon, length_of_sequences, start_of_sequences_test)
    X_test = pad_sequences(X_test, padding='post', dtype='float32')

    return X_train, X_test, y_train, y_test, y_b_test, y_b_train


""" MODEL """
def init_model():

  model = Sequential()
  model.add(layers.Masking())
  model.add(layers.LSTM(7, activation='tanh'))
  model.add(layers.Dense(10, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])# find loss an penalizing negative or positive results - alternatives to mae
                                # mae weighted
  return model


def model_fitting(model, X_train, y_train):

  model = init_model()

  es = EarlyStopping(patience=40, mode='min', verbose=1, monitor='val_loss',restore_best_weights=True)
  # patience = 30 - > if my algorithm doesnt improve my performance for a certain number of iterations
  # (patience) stop the fitting process, return the weights as they are

  history = model.fit(X_train,y_train,
            batch_size=16,
            epochs=10000, # iteration through your data
            validation_split=0.2,
            callbacks=[es],
            verbose=1)

  return history


def model_evaluation(model, X_test, y_test):

  y_pred = model.predict(X_test)
  res = model.evaluate(X_test,y_test)
  print(f'Loss on the test set : {res[0]:.8f}')
  print(f'Accuracy on the test set : {res[1]:.8f}')

  return y_pred, res


if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import bitcoin_price_predictor
    folder_source, _ = split(bitcoin_price_predictor.__file__)
    # data = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    clean_data = clean_data(data)
    print(' dataframe cleaned')

