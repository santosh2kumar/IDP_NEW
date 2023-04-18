FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas

COPY train_test_preprocess.py ./train_test_preprocess.py

ENTRYPOINT [ "python", "train_test_preprocess.py" ]
