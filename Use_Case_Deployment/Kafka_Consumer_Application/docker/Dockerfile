FROM python:3.7-slim

WORKDIR /app

RUN pip install -U numpy pandas requests redis-py-cluster kafka-python cassandra-driver

COPY KafkaConsumerCassandra.py ./KafkaConsumerCassandra.py

ENTRYPOINT [ "python", "KafkaConsumerCassandra.py" ]

