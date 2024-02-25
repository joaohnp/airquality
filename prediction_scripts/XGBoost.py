#   %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from utils import create_features, querying_sql

DATABASE_URI = 'postgresql+psycopg2://master:amsterdam@localhost:5432/airquality'

# %%
query = querying_sql(DATABASE_URI, 2019, 3)
data = []
for record in query:
    timestamp = pd.Timestamp(year=record.year, month=record.month, day=record.day, hour=record.hour)
    data.append({'Timestamp': timestamp, 'Measurement': record.measurement})

df_measurements = pd.DataFrame(data)
df_measurements.set_index('Timestamp', inplace=True)
#%%
df_measurements["Measurement"].plot(title='PM10 values across the year')
dataset = df_measurements["Measurement"]
# %% Train/test
train_size = int(len(dataset)*0.8)
train, test = dataset[:train_size], dataset[train_size:]

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
# Assuming y_test are your actual values and y_pred are your model's predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='blue', alpha=0.2)
plt.plot(y_pred, label='Predicted', color='red', linestyle='--',alpha=0.7)
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Hourly measurments')
plt.ylabel('PM10')
plt.title(f'XGBoost prediction RMSE = {"%.2f" % rmse}')
plt.legend()
# %%
