https://data.rivm.nl/data/luchtmeetnet/Vastgesteld-jaar/2013/

docker run --name airquality -e POSTGRES_USER=master -e POSTGRES_PASSWORD=amsterdam -e POSTGRES_DB=mydatabase -p 5432:5432 -d postgres

https://drawsql.app/teams/teste-124/diagrams/airquality

Predicting air quality in Amsterdam

This project is a showcase of a machine learning problem/pipeline. Here we'll go from creating a postgres server using docker, populating and reading into our 3 selected models (xgboost, rnn, cnn). After that, we'll run pycaret to test which model is the best