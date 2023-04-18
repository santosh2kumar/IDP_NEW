FROM python:3.7-slim

WORKDIR /app

RUN pip install -U numpy pandas kafka-python

COPY KafkaProducer.py ./KafkaProducer.py
COPY debug.py ./debug.py

ENTRYPOINT [ "python", "KafkaProducer.py" ]
