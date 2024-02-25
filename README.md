
# Predicting air quality

Simple tool to showcase end-to-end data science. 

Here the goal is to get stablished data, build a postgres server with it and use such data to predict air quality in the next hour, by using different popular machine/deep learning solutions: XGBoost, RNN and CNN. 

We'll be using data collected by the dutch government on air quality; specifically PM10 particles. 

Here is where we got our dataset: https://data.rivm.nl/data/luchtmeetnet/Vastgesteld-jaar/

This is the schema designed for the SQL database: https://drawsql.app/teams/teste-124/diagrams/airquality


## Run Locally

Clone the project

```bash
  git clone https://github.com/joaohnp/airquality/
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requeriments.txt
```

Start the postgres server

```bash
  cd docker
  docker compose up
```

Mine the data

```bash
  cd ..
  python mining.py
```
Check one of the solutions:

```bash
  cd prediction_scripts
  python XGBoost.py
```


## Tech Stack

**Data storage and management:** PostgresSQL, Docker

**Data preparation and analysis:** pandas, SQLalchemy

**Machine Learning and Deep Learning:** Keras, TensorFlow, XGBoost


## Authors

- [@joaohnp](https://www.github.com/joaohnp)


## Support

For support, email joaohnp@gmail.com
