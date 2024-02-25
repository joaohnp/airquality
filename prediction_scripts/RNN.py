#   %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from utils import create_features, querying_sql

DATABASE_URI = 'postgresql+psycopg2://master:amsterdam@localhost:5432/airquality'

query = querying_sql(DATABASE_URI, 2019, 3)
data = []
for record in query:
    timestamp = pd.Timestamp(year=record.year, month=record.month, day=record.day, hour=record.hour)
    data.append({'Timestamp': timestamp, 'Measurement': record.measurement})

df_measurements = pd.DataFrame(data)
df_measurements.set_index('Timestamp', inplace=True)
df_measurements["Measurement"].plot(title='PM10 values across the year')
dataset = df_measurements["Measurement"]
# %% Train/test
train_size = int(len(dataset)*0.8)
train, test = dataset[:train_size], dataset[train_size:]
LAG = 12
X_train, y_train = create_features(train, LAG)
X_test, y_test = create_features(test, LAG)


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Fit model
model.fit(X_train, y_train, epochs=100, batch_size=32)
# Generate predictions for the test data
y_pred = model.predict(X_test)
#%%
# Ensure y_pred and y_test are of the same shape, especially if one is a DataFrame/Series
y_pred = y_pred.flatten()  # Flatten y_pred if it's 2D
y_test = y_test.flatten()  # Ensure y_test is also flat if it's not already
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='blue', alpha=0.2)
plt.plot(y_pred, label='Predicted', color='red', linestyle='--',alpha=0.7)
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Hourly measurments')
plt.ylabel('PM10')
plt.title(f'RNN prediction RMSE = {"%.2f" % rmse}')
plt.legend()


# %%
