FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas sklearn

COPY rf_train.py ./rf_train.py

ENTRYPOINT [ "python", "rf_train.py" ]
