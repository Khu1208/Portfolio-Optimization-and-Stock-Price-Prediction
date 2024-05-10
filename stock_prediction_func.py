import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import GRU, Dense
import plotly.graph_objects as go
import streamlit as st
from pypfopt import EfficientFrontier
from pypfopt import risk_models


def fetch_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data
def create_dataset(data_array, timesteps):
   data_array = data_array.to_numpy()
   X_3d = np.array([data_array[i-timesteps:i, 0:] for i in range(timesteps, len(data_array))])
   y_3d = np.array([data_array[i,0 :] for i in range(timesteps, len(data_array))])
   return X_3d, y_3d

def lstm_model(X_train, y_train, X_test, y_test):
    # Assuming X_train and y_train are your input and output data
    # Check data types and convert if needed
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(30, 10)))  # Assuming 30 time steps and 10 features
    model.add(Dense(units=10, activation='linear'))  # Output layer with 10 units for 10 companies using linear activation
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Once the model is trained, you can use it to make predictions
    predictions_lstm = model.predict(X_test)  # Assuming X_test has the same shape as X_train
    return predictions_lstm

def rnn_model(X_train, y_train, X_test, y_test):
    # Define the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=64, input_shape=(30, 10)))  # Assuming 30 time steps and 10 features
    model.add(Dense(units=10, activation='linear'))  # Output layer with 10 units for 10 companies using linear activation
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Once the model is trained, you can use it to make predictions
    predictions_rnn = model.predict(X_test)  # Assuming X_test has the same shape as X_train
    return predictions_rnn

def cnn_model(X_train, y_train, X_test, y_test):
    # Define the CNN model
    # Reshape the data for CNN input
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(30, 10,1)))  # Assuming 30 time steps and 10 features
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='linear'))  # Output layer with 10 units for 10 companies using linear activation
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X_train_cnn, y_train, epochs=50, batch_size=32)

    # Once the model is trained, you can use it to make predictions
    predictions_cnn = model.predict(X_test_cnn)  # Assuming X_test has the same shape as X_train
    return predictions_cnn

def gru_model(X_train, y_train, X_test, y_test):
    # Define the GRU model
    model = Sequential()
    model.add(GRU(units=64, input_shape=(30, 10)))  # Assuming 30 time steps and 10 features
    model.add(Dense(units=10, activation='linear'))  # Output layer with 10 units for 10 companies using linear activation
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Once the model is trained, you can use it to make predictions
    predictions_gru = model.predict(X_test)  # Assuming X_test has the same shape as X_train
    return predictions_gru

import plotly.express as px
import numpy as np

def calculate_error(y_test,predictions):
  mse_per_company = []
  for i in range(10):
    mse_company = mean_squared_error(y_test[:, i], predictions[:, i])
    mse_per_company.append(mse_company)
  return mse_per_company


def plot_mse_comparison(companies, lstm_mse, rnn_mse, cnn_mse, gru_mse):
    bar_width = 0.2
    index = list(range(len(companies)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=index, y=lstm_mse, width=bar_width, name='LSTM'))
    fig.add_trace(go.Bar(x=[i + bar_width for i in index], y=rnn_mse, width=bar_width, name='RNN'))
    fig.add_trace(go.Bar(x=[i + 2*bar_width for i in index], y=cnn_mse, width=bar_width, name='CNN'))
    fig.add_trace(go.Bar(x=[i + 3*bar_width for i in index], y=gru_mse, width=bar_width, name='GRU'))
    
    fig.update_layout(width = 1200, xaxis={'tickvals': [i + 1.5*bar_width for i in index], 'ticktext': companies},
                      xaxis_tickangle=-45,
                      yaxis_title='Mean Squared Error (MSE)',
                      title='Comparison of MSE for Different Models',
                      barmode='group')
    return fig
