"""Utils with needed functions to query and create features.
    """
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Measurement


def create_features(data, lag=1):
    """This function generates features to predict PM10 level in the next hour

    Parameters
    ----------
    data : float
        PM10 value per hour
    lag : int, optional
        number of measurements to predict the next one, by default 1

    Returns
    -------
    numpy array
        returns a numpy array with the features
    """
    X, y = [], []
    # Adjust the range to stop before the last `lag` elements
    for i in range(len(data) - lag):
        # Ensure indexing works by converting to a list or directly accessing values
        X.append(data.iloc[i:(i + lag)].values.tolist())  # Using .iloc for DataFrame or Series
        y.append(data.iloc[i + lag])
    return np.array(X), np.array(y)


def querying_sql(DBURI:"URI", year:int, location_id:int):
    """receives db and filters year and location of measurement

    Parameters
    ----------
    dbURI : url
        db location
    year : int
        year of interest
    location_id : code for measurement of PM10
        which station id

    Returns
    -------
    pd.Series
        returns all the measurments for that filter
    """
    # Database connection string
    database_uri = DBURI

    engine = create_engine(database_uri)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(
        Measurement.location_id,
        Measurement.day,
        Measurement.month,
        Measurement.year,
        Measurement.hour,
        Measurement.measurement
    ).filter(
        Measurement.year == year,
        Measurement.location_id == location_id
    ).all()

    return query