# Portfolio-Optimization-and-Stock-Price-Prediction
Applied machine learning model to predict stock price and mean-variance portfolio optimization

# Financial Mathematics and Machine Learning for Portfolio Optimization

Welcome to the Financial Mathematics and Machine Learning project for portfolio optimization. This project combines financial concepts like rate of return, portfolio weights, beta, efficient frontier, mean-variance optimization, and machine learning models such as RNN, LSTM, GRU, and CNN for stock price prediction and portfolio management.

## Rate of Return

The rate of return (`RT`) between two time points `t0` and `t1` can be calculated using the formula:

$\[ RT = \frac{P_{t1} - P_{t0}}{P_{t0}} \times 100 \]$

## Portfolio Weights

Portfolio weights (`Wi`) indicate the proportion of capital invested in individual assets comprising a portfolio.

## Beta

Beta (`𝜷`) measures the volatility of an asset compared to the overall market. It can be calculated using the formula:

$\[ 𝜷 = \frac{cov(ri, rP)}{var(rP)} \]$

## Portfolio Beta

The portfolio beta (`𝜷i`) is calculated as the weighted sum of individual betas (`Wi`) of assets in the portfolio.

## Portfolio Return

The portfolio return (`R`) is the weighted sum of returns of individual assets in the portfolio.

## Portfolio Risk

Portfolio risk (`𝞼ij`) is calculated using the covariance matrix between returns of assets `i` and `j`.

## Markowitz’s Mean-Variance Optimization Model

Markowitz's Mean-Variance Optimization model aims to maximize returns while minimizing risk, subject to certain constraints such as portfolio weights and expected return.

### Objectives

- Maximize expected portfolio return (`rx`).
- Minimize portfolio variance (`PV`).
- Ensure portfolio weights sum to 1.

### Efficient Frontier

The Efficient Frontier graphically represents optimal portfolios with varying risk-return profiles.

## Machine Learning Models

### RNN (Recurrent Neural Network)

RNN is a neural network architecture suitable for sequential data processing.

### LSTM (Long Short-Term Memory)

LSTM is a variant of RNN designed to handle long-term dependencies in sequences.

### GRU (Gated Recurrent Unit)

GRU is another variant of RNN with a simplified structure for capturing sequential patterns.

### CNN (Convolutional Neural Network)

CNN is suitable for processing grid-like data such as images or time-series data.

## Data Collection and Preprocessing

Data is collected using the `yfinance` library, normalized, and converted into sequences for prediction tasks.

## Portfolio Optimization

The project includes implementations of Markowitz's Mean-Variance Optimization model for portfolio optimization. It aims to find the optimal weights for assets to maximize returns and minimize risk.

## Future Work

- Implement advanced portfolio optimization models such as Mean-VaR, Mean-CVaR, and Mean-EVaR.
- Compare and evaluate different machine learning models for stock price prediction.
- Enhance data collection and preprocessing techniques.

## Contributors

- [Khushali Nandani][(link-to-contributor-profile) ](https://github.com/Khu1208)
- [Bhumika Gupta](link-to-contributor-profile) 
