#   %%
import os

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% had to detect encoding - pandas threw errors. 
dir_path = os.path.dirname(os.path.realpath(__file__))
src_PM10 = os.path.join(dir_path, 'src', 'PM10')
all_filesPM10 = os.listdir(src_PM10)
city_of_interest = 'Amsterdam'
# Creating an empty dictionary with keys as the year of interest
year_dict = {}


#   %%
for file_name in all_filesPM10:
    with open(os.path.join(src_PM10, file_name), 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        PM10 = pd.read_csv(os.path.join(src_PM10, file_name), encoding =encoding, sep=';')  # Example for semicolon-delimited files
        pd.to_datetime(PM10["Unnamed: 3"], errors='coerce', format='%Y%m%d %H:%M')
        
        contains_amsterdam = PM10.iloc[1].str.contains(city_of_interest, case=False, na=False)
        matching_columns = PM10.columns[contains_amsterdam]
        PM10_filtered = PM10[matching_columns]
        year = file_name.split("_")[0]
        year_dict[year] = pd.concat([PM10["Unnamed: 3"][9:-1], PM10_filtered[9:-1]], axis=1)


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


def create_features(data, lag=1):
    X, y = [], []
    # Adjust the range to stop before the last `lag` elements
    for i in range(len(data) - lag):
        # Ensure indexing works by converting to a list or directly accessing values
        X.append(data.iloc[i:(i + lag)].values.tolist())  # Using .iloc for DataFrame or Series
        y.append(data.iloc[i + lag])
    return np.array(X), np.array(y)


lag = 12  # For example, use the last 24 hours to predict the next hour
X_train, y_train = create_features(train, lag)
X_test, y_test = create_features(test, lag)

nan_in_y_train = np.isnan(y_train).sum()
nan_in_y_test = np.isnan(y_test).sum()

#%%
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Initialize the model
model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5)

# Fit the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
#%%
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {"objective": "reg:squarederror", 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500}

cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10,
                    metrics="rmse", as_pandas=True, seed=123)
#%%
import matplotlib.pyplot as plt

# Assuming y_test are your actual values and y_pred are your model's predictions
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
plt.plot(y_test, label='Actual', color='blue', alpha=0.2)  # Plot actual values
plt.plot(y_pred, label='Predicted', color='red', linestyle='--') # Plot predicted values
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Index')  # Adjust as appropriate (e

# %%
