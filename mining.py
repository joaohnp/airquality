"""Module to read the csv files and populate the database
"""
#%%
import os

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# %% Connecting to the database. 
DATABASE_URI = 'postgresql+psycopg2://master:amsterdam@localhost:5432/airquality'
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()


# %%
Base = declarative_base()

class Location(Base):
    """Class to populate the 'location' table

    Parameters
    ----------
    Base : declarative_base
        id, code, city
    """
    __tablename__ = 'locations'
    location_id = Column(Integer, primary_key=True)
    code = Column(String)
    city = Column(String)
    # Measurements relationship
    measurements = relationship("Measurement", back_populates="location")

class Measurement(Base):
    """Class to populate the 'measurements' table

    Parameters
    ----------
    Base : declarative base
        measurementid, location_id, day, month, year, measurement
    """
    __tablename__ = 'measurements'
    measurementid = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey('locations.location_id'))
    day = Column(Integer)
    month = Column(Integer)
    year = Column(Integer)
    hour = Column(Integer)
    measurement = Column(Float)
    # Relationship to Location
    location = relationship("Location", back_populates="measurements")

# %% Now, reading the csv files

dir_path = os.path.dirname(os.path.realpath(__file__))
src_PM10 = os.path.join(dir_path, 'src', 'PM10')
all_filesPM10 = os.listdir(src_PM10)
CITY_OF_INTEREST = ["Amsterdam", "Rotterdam"]
# Creating an empty dictionary with keys as the year of interest
year_dict = {city: {} for city in CITY_OF_INTEREST}


#   %%
for city in CITY_OF_INTEREST:
    for file_name in all_filesPM10:
        with open(os.path.join(src_PM10, file_name), 'rb') as file:
            raw_data = file.read()
            PM10 = pd.read_csv(os.path.join(src_PM10, file_name),
                            encoding ='ISO-8859-1', sep=';')
            PM10.rename(columns={"Unnamed: 3": "Timestamp"}, inplace=True)
            # Example for semicolon-delimited files
            PM10["Timestamp"] = pd.to_datetime(PM10["Timestamp"], errors='coerce',
                        format='%Y%m%d %H:%M')
            contains_city = PM10.iloc[1].str.contains(city,case=False, na=False)
            PM10_filtered = PM10[PM10.columns[contains_city]]
            year = file_name.split("_")[0]
            year_dict[city][year] = pd.concat([PM10["Timestamp"][9:-1],
                                        PM10_filtered[9:-1]],
                                        axis=1).reset_index(drop=True)
            for col in year_dict[city][year].columns[1:-1]:
                year_dict[city][year][col] = pd.to_numeric(year_dict[city][year][col],
                                                    errors='coerce')
            year_dict[city][year] = year_dict[city][year].ffill()
#%% Inserting in the database
# Path to your Excel file(s)
cities = year_dict.keys()
for city in cities:
    years = year_dict[city].keys()
    for year in years:
        # Read the Excel file into a pandas DataFrame
        df = year_dict[city][year]
        locations = df.columns[1:-1]
        for location_code in locations:

            location = Location(code=location_code, city=city)
            session.add(location)
            session.commit()

            # df_measures = year_dict[city][year][location_code]

            for index, row in df.iterrows():
                timestamp = row['Timestamp']
                day = timestamp.day
                month = timestamp.month
                year = timestamp.year
                hour = timestamp.hour
                measurement_value = row[location_code]
                # Now deal with measurements
                measurement = Measurement(
                    location_id=location.location_id,
                    day=day,
                    month=month,
                    year=year,
                    hour=hour,
                    measurement=measurement_value
                )
                session.add(measurement)

            # Commit once per file or per batch of rows to optimize
            session.commit()
# After all database operations are done
session.close()
