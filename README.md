# Stock-Price-Prediction
This repository contains code for a stock price prediction program built using machine learning algorithms. The model uses historical stock data from Yahoo Finance to predict future stock prices. It is designed to predict buy, sell, or hold signals based on past stock price movements, using technical indicators as features.

Features

Data Source: Yahoo Finance API for historical stock data.
Technical Indicators: Uses 3 technical indicators along with OHLCV data to predict next closing price of a particular stock
Machine Learning Model: The model uses a combination of CNN (Convolutional Neural Networks) and RNN (Recurrent Neural Networks) to predict the next stock price movement.
Predictions: Outputs predicts the closing price
Sliding Window: A time-series sliding window approach to feed sequences of past prices into the model for predictions for temporal dependencies
