# importing libraries

import streamlit as st
import pandas as pd
import yfinance as yf
# from datetime import datetime,timedelta
import datetime
import m1varfunc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import stock_prediction_func
st.set_page_config(page_title = "Prediction and optimization",
    layout = 'wide')

st.title("Stock price prediction")

companies = ['INFY.NS', 'TCS.NS', 'TATAMOTORS.NS', 'MARUTI.NS',
                 'SUNPHARMA.NS', 'CIPLA.NS', 'ITC.NS', 'MARICO.NS', 'GOLDBEES.NS', 'BAJAJ-AUTO.NS']

@st.cache_data()
def fetch_and_preprocess_stock_data():
    companies = ['INFY.NS', 'TCS.NS', 'TATAMOTORS.NS', 'MARUTI.NS',
                 'SUNPHARMA.NS', 'CIPLA.NS', 'ITC.NS', 'MARICO.NS', 'GOLDBEES.NS', 'BAJAJ-AUTO.NS']
    end_date = datetime.date.today()
    start_date = datetime.date(datetime.date.today().year - 10, datetime.date.today().month, datetime.date.today().day)
    stock_data = stock_prediction_func.fetch_stock_data(companies, start_date, end_date)
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(stock_data), columns=stock_data.columns, index=stock_data.index)
    df_new = pd.DataFrame(normalized_data)
    df = df_new.dropna()
    print(df.shape)

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # Display the head and tail of the DataFrame
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Dataframe head")
        st.dataframe(train_data.head(), use_container_width=True)
    with col2:
        st.markdown("### Dataframe tail")
        st.dataframe(test_data.tail(), use_container_width=True)
    
    return train_data, test_data

# Fetch, preprocess, and split stock data
train_data, test_data = fetch_and_preprocess_stock_data()

# Define a function to train the prediction models
@st.cache_data
def train_models(train_data):
    look_back = 30  # Number of previous time steps to consider
    X_train, y_train = stock_prediction_func.create_dataset(train_data, look_back)
    
    # Define X_test and y_test using the test data
    X_test, y_test = stock_prediction_func.create_dataset(test_data, look_back)
    print(X_train.shape, X_test.shape)
    prediction_lstm = stock_prediction_func.lstm_model(X_train, y_train, X_test, y_test)
    prediction_rnn = stock_prediction_func.rnn_model(X_train, y_train, X_test, y_test)
    prediction_cnn = stock_prediction_func.cnn_model(X_train, y_train, X_test, y_test)
    prediction_gru = stock_prediction_func.gru_model(X_train, y_train, X_test, y_test)
    mse_lstm = stock_prediction_func.calculate_error(y_test, prediction_lstm)
    mse_cnn = stock_prediction_func.calculate_error(y_test, prediction_cnn)
    mse_rnn = stock_prediction_func.calculate_error(y_test, prediction_rnn)
    mse_gru = stock_prediction_func.calculate_error(y_test, prediction_gru)
    return mse_lstm, mse_rnn, mse_cnn, mse_gru

# Train the prediction models
mse_lstm, mse_rnn, mse_cnn, mse_gru = train_models(train_data)
st.plotly_chart(stock_prediction_func.plot_mse_comparison(companies,mse_lstm,mse_rnn,mse_cnn,mse_gru))
st.title("Mean variance optimization")
# getting input from user

def second_portion():
    col1, col2 = st.columns([1,1])
    with col1:
        stocks_list = st.multiselect("Choose stocks", ('INFY.NS','TCS.NS','TATAMOTORS.NS','MARUTI.NS',
	    'SUNPHARMA.NS','CIPLA.NS','ITC.NS','MARICO.NS','GOLDBEES.NS','BAJAJ-AUTO.NS'),['INFY.NS' , 'CIPLA.NS', 'TATAMOTORS.NS','ITC.NS'])
    with col2:
        year = st.number_input("Number of years",1,10)

# try:
    start = datetime.date(datetime.date.today().year-year, datetime.date.today().month, datetime.date.today().day)
    end = datetime.date.today()

    stocks_df = pd.DataFrame()

    for stock in stocks_list:
        data = yf.download(stock, period = f'{year}y')
        stocks_df[f'{stock}'] = data['Close']

    print(stocks_df.head())

    stocks_df.reset_index(inplace = True)

    stocks_df['Date'] = stocks_df['Date'].apply(lambda x:str(x)[:10])
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(m1varfunc.plot_trends(stocks_df))
    with col2:
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
        stocks_df.set_index('Date', inplace=True)
        st.markdown("### Variance-Covariance matrix")
        cov_mtrx = m1varfunc.covariance_matrix(stocks_df)
        st.dataframe(cov_mtrx, use_container_width = True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Heatmap of covariance matrix")
        st.plotly_chart(m1varfunc.heatmap_covar(cov_mtrx))
    with col2:
        returns=m1varfunc.return_capm(stocks_df)
        st.markdown("### Expected returns")
        st.dataframe(returns, use_container_width = True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Weights at Maximum Sharpe")
        weights_max_sharpe,performance_maxsharpe= m1varfunc.max_sharpe_weights(returns, cov_mtrx)
        st.write("Expected annual return, volatility, and Sharpe ratio:", performance_maxsharpe)
        st.dataframe(weights_max_sharpe, use_container_width = True)

    with col2:
        st.markdown("### Weights at Minimum Variance")
        weights_min_variance,performance_minvariance= m1varfunc.min_variance_weights(returns, cov_mtrx)
        st.write("Expected annual return, volatility, and Sharpe ratio:", performance_minvariance)
        st.dataframe(weights_min_variance, use_container_width = True)

second_portion()
# except:
#     st.write("Error in fetching data. Please try again")