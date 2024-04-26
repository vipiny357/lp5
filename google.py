import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("google.csv")
data.head(5)



y = data["close"].values.reshape(-1, 1)
y



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(y)
scaled_data



train_size = int(len(scaled_data)*0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]



def create_sequence(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i]+seq_length)

    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequence(train_data, seq_length)
X_test, y_test = create_sequence(test_data, seq_length)



import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM



model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)



model.evaluate(X_test, y_test)



plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])