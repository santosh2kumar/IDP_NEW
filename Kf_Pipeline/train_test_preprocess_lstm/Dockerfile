FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas sklearn

COPY preprocess_train_test_lstm.py ./preprocess_train_test_lstm.py

ENTRYPOINT [ "python", "preprocess_train_test_lstm.py" ]
