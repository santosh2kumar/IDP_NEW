FROM python:3.7-slim

WORKDIR /app

RUN pip install -U numpy pandas kafka-python confluent-kafka avro requests

COPY fraud_producer_avro.py ./fraud_producer_avro.py

ENTRYPOINT [ "python", "fraud_producer_avro.py" ]
