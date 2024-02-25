"""Storing classes to be used by SQLAlchemy
"""
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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