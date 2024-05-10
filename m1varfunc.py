import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.plotting import plot_weights
from pypfopt.cla import CLA
from pypfopt import plotting
from pypfopt import objective_functions

def plot_trends(stock_data):
    # Melt the DataFrame to have 'Date', 'Company', and 'Price' columns
    melted_data = stock_data.melt(id_vars='Date', var_name='Company', value_name='Price')

# Plotting the stock price trend
    fig = px.line(melted_data, x='Date', y='Price', color='Company', 
              title='Stock Price Trend for Multiple Companies')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Stock Price')
    return fig

def covariance_matrix(portfolio):
    sample_cov = risk_models.sample_cov(portfolio, frequency=252)
    return sample_cov

def heatmap_covar(cov_matrix):
    # Plot heatmap
    fig = px.imshow(cov_matrix,
                labels=dict(color="Correlation"),
                x=cov_matrix.index,
                y=cov_matrix.columns,
                title="Correlation Heatmap of Portfolio Stocks",
                color_continuous_scale="Viridis")

    return fig

def return_capm(portfolio):
    mu = expected_returns.capm_return(portfolio)
    return mu

def max_sharpe_weights(mu, sigma):
    # Maximum Sharpe Ratio
    ef = EfficientFrontier(mu, sigma)
    raw_weights_maxsharpe_exp = ef.max_sharpe(risk_free_rate=0.05)

    performance_maxsharpe=ef.portfolio_performance(verbose = True, risk_free_rate = 0.05)
    return raw_weights_maxsharpe_exp,performance_maxsharpe

def min_variance_weights(mu, sigma):
    # Minimum Variance
    ef = EfficientFrontier(mu, sigma)
    raw_weights_minvar_exp = ef.min_volatility()

    performance_minvariance=ef.portfolio_performance(verbose = True, risk_free_rate = 0.05)
    return raw_weights_minvar_exp,performance_minvariance



