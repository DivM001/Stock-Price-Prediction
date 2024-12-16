import numpy as np , pandas as pd, matplotlib.pyplot as plt
import sklearn as sk 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import yfinance as yf 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf 
import pandas_ta as ta , mplfinance as mpl 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Convolution1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from keras import callbacks

#Getting the data
ticker = "TATAMOTORS.NS"
pricedata = yf.download(ticker, start='2022-01-01', auto_adjust=True)
plt.plot(pricedata['Close'])

# Stock Information
stock = yf.Ticker(ticker)
info = stock.info
sector =info.get('sector')
industry = info.get('industry')
print(industry)
print(sector)
pe = info.get('forwardPE')
revenue = info.get('totalRevenue', 'N/A')
net_profit = info.get('netIncomeToCommon', 'N/A')
print(pe)
print(revenue)
print(net_profit)



# Calculating RSI
pricedata['RSI'] = ta.rsi(pricedata['Close'], length= 14)
pricedata.dropna(subset=['RSI'], inplace = True)
#Calculating OBV
pricedata['OBV'] = ta.obv(pricedata['Close'], pricedata['Volume'])



# Calculate Stochastic Oscillator (%K and %D)
stoch = ta.stoch(pricedata['High'], pricedata['Low'], pricedata['Close'], 14, 3, 1)
pricedata["stoch_K"] = stoch['STOCHk_14_3_1']
pricedata["stoch_D"] = stoch['STOCHd_14_3_1']

#Bollinger Bands
bb = ta.bbands(pricedata['Close'], length = 20, std =2, ddof =0, mamode='ema',talib=False)
pricedata= pricedata.join(bb)

scaler = StandardScaler()


# Extract features (assuming all columns except 'Close' are features)
X = pricedata.drop(columns=['Close']).values
# Extract and scale target variable
y = pricedata['Close'].values

# Scale the features and target separately
scaled_X = scaler.fit_transform(X)
scaled_y = scaler.fit_transform(y.reshape(-1, 1))

# Step 2: Create sequences of `window_size` for the features (X)
window_size = 10
X_reshaped = []

# Create sliding windows for the features
for i in range(window_size, len(scaled_X)):
    X_reshaped.append(scaled_X[i - window_size:i])  # Create a sequence of `window_size` timesteps

X_reshaped = np.array(X_reshaped)

# Step 3: Align `y` with `X_reshaped`
# `y` should start from `window_size` as we're predicting the value after each window
y = scaled_y[window_size:]

# Step 4: Remove NaNs (if any) from X_reshaped and y
# Apply NaN filtering to both X_reshaped and y at the same time
mask = ~np.isnan(X_reshaped).any(axis=(1, 2))  # Check if there are any NaNs in the feature array
X_reshaped = X_reshaped[mask]
y = y[mask]

# Step 5: Train-Test-Validation Split
# Split the reshaped data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_reshaped, y, test_size=0.2, shuffle=False)

# Split the temporary data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)


model = Sequential([
    # Conv1D Layer: Extract features from the time series data
    Convolution1D(filters=5, kernel_size=5, activation='relu', input_shape=(10, 13)),  # 10 timesteps, 13 features
    
    # GRU Layer: Capture temporal dependencies (set return_sequences=True to pass sequences to LSTM)
    GRU(units=15, activation='relu', return_sequences=True ),  # Ensure output is a sequence for LSTM
    
    # Dropout Layer: Regularization to avoid overfitting
    Dropout(0.4),
    
    # LSTM Layer: Capture long-term dependencies
    LSTM(units=7,input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.01)), 
    
    # Output Layer: Predict the next stock price (regression)
    Dense(units=1, activation='linear')  # Regression output (next price)
])


print(model.summary())

# Compile the model
model.compile(optimizer=Adam(learning_rate =0.0001), loss=MeanSquaredError, metrics=[MeanAbsoluteError()])


# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss, MeanAbsoluteError= model.evaluate(X_test, y_test) 
print(f"Test Loss: {loss}")
print(MeanAbsoluteError)

y_pred = model.predict(X_val)

# Calculate additional metrics
mse = np.mean((y_val - y_pred)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_val - y_pred))

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')



train_loss = history.history['loss']  # Training loss history
val_loss = history.history['val_loss']  # Validation loss history

# Print the last epoch's training and validation loss (j(train) and j(val))
print("Last epoch training loss (j(train)):", train_loss[-1])
print("Last epoch validation loss (j(val)):", val_loss[-1])



