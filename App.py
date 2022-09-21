import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# We should take bigger timeframe for our model to be accurate
start = '2010-01-01'
end = '2022-07-16'

# User can take any stock picker
st.title('Stock Price Prediction ')
user_input = st.text_input('Enter Stock Ticker', 'AMZN')
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing data
st.subheader('Data from 2010  -  2022')
st.write(df.describe())

# Visualisations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart  :\n100 day Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart  :\n100 Days Moving Average and 200 Days Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
# We take 70% data for training and 30% for testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

# Now we should move on to scaling down data for LSTM model
# We have to scale down it between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# scaler.fit_transform return an array
data_training_array = scaler.fit_transform(data_training)

# Load my model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)