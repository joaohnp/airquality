#   %%
import os

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from utils import create_features

#   %% had to detect encoding - pandas threw errors. 
dir_path = os.path.dirname(os.path.realpath(__file__))
src_PM10 = os.path.join(dir_path, 'src', 'PM10')
all_filesPM10 = os.listdir(src_PM10)
CITY_OF_INTEREST = 'Amsterdam'
# Creating an empty dictionary with keys as the year of interest
year_dict = {}


#   %%
for file_name in all_filesPM10:
    with open(os.path.join(src_PM10, file_name), 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        PM10 = pd.read_csv(os.path.join(src_PM10, file_name),
                           encoding =encoding, sep=';')  
        # Example for semicolon-delimited files
        pd.to_datetime(PM10["Unnamed: 3"], errors='coerce',
                       format='%Y%m%d %H:%M')
        contains_amsterdam = PM10.iloc[1].str.contains(CITY_OF_INTEREST, 
                                                       case=False, na=False)
        matching_columns = PM10.columns[contains_amsterdam]
        PM10_filtered = PM10[matching_columns]
        year = file_name.split("_")[0]
        year_dict[year] = pd.concat([PM10["Unnamed: 3"][9:-1], 
                                     PM10_filtered[9:-1]], axis=1)


# Read CSV with detected encoding
#   %%
toplot = year_dict["2022"]["NL49012"]
cleaned = toplot.fillna(method='ffill')
cleaned2 = pd.to_numeric(cleaned)
plt.plot(cleaned2)
# %% Train/test
train_size = int(len(cleaned2)*0.8)
train, test = cleaned2[:train_size], cleaned2[train_size:]

#   %%
LAG = 12  # For example, use the last 24 hours to predict the next hour
X_train, y_train = create_features(train, LAG)
X_test, y_test = create_features(test, LAG)

nan_in_y_train = np.isnan(y_train).sum()
nan_in_y_test = np.isnan(y_test).sum()

# Initialize the model
model = xgb.XGBRegressor(objective ='reg:squarederror', 
                         n_estimators=1000, learning_rate=0.05, max_depth=5)

# Fit the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
#   %%
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {"objective": "reg:squarederror", 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500}

cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10,
                    metrics="rmse", as_pandas=True, seed=123)
#%%

# Assuming y_test are your actual values and y_pred are your model's predictions
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
plt.plot(y_test, label='Actual', color='blue', alpha=0.2)  # Plot actual values
plt.plot(y_pred, label='Predicted', color='red', linestyle='--',alpha=0.7) # Plot predicted values
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Index')  # Adjust as appropriate (e

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% Trying the same thing, but for ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

result = adfuller(train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# Since the data is confidently stationary, d =0
plot_acf(train)
plot_pacf(train)
plt.show()








# Assume `data` is your time series data and is stationary
model = ARIMA(test)
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=24, order=(40, 0, 5))  # Forecasting next 5 steps for example
plt.plot(forecast)
model_fit.plot_diagnostics(figsize=(10, 8))
plt.show()



# %%

# Assuming `X_train` and `y_train` are prepared
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

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='.', linestyle='-', linewidth=1.5, color='blue')
plt.plot(y_pred, label='Predicted', marker='.', linestyle='--', linewidth=1.5, color='red')
plt.title('LSTM Forecast vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Observation Value')
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
#%%
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(12, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# %%
# Generate predictions
y_pred = model.predict(X_test)

# Assuming data scaling was applied, inverse transform the predictions and actual values if necessary

# Calculate RMSE for evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred.flatten()))
print("RMSE: %f" % (rmse))

# Plotting the actual vs predicted values for visual comparison
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='.', linestyle='-', linewidth=1.5, color='blue')
plt.plot(y_pred.flatten(), label='Predicted', marker='.', linestyle='--', linewidth=1.5, color='red')
plt.title('CNN Forecast vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Observation Value')
plt.legend()
plt.show()

# %%
