#   %%
import os

import chardet
import pandas as pd

# %% had to detect encoding - pandas threw errors. 
dir_path = os.path.dirname(os.path.realpath(__file__))
src_PM10 = os.path.join(dir_path, 'src', 'PM10')

with open(os.path.join(src_PM10, '2014_PM10.csv'), 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# Read CSV with detected encoding
PM10 = pd.read_csv(os.path.join(src_PM10, '2014_PM10.csv'), encoding =encoding, sep=';')  # Example for semicolon-delimited files
# %% Finding columns for the city of interest
city_of_interest = 'Amsterdam'


# Find columns that contain 'Amsterdam'
contains_amsterdam = PM10.iloc[1].str.contains(city_of_interest, case=False, na=False)

# Select column names that contain 'Amsterdam'
matching_columns = PM10.columns[contains_amsterdam]

print("Columns containing 'Amsterdam':")
print(matching_columns)
PM10_filtered = PM10[matching_columns]
# %%
