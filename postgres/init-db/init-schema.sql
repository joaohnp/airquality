-- init-schema.sql

CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    code TEXT NOT NULL,
    city TEXT NOT NULL
);

CREATE TABLE measurements (
    measurementID SERIAL PRIMARY KEY,
    location_id INTEGER NOT NULL REFERENCES locations(location_id),
    day INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    measurement FLOAT NOT NULL
);
