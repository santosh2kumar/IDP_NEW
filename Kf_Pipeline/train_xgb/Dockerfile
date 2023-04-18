FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas sklearn xgboost

COPY xgb_train.py ./xgb_train.py

ENTRYPOINT [ "python", "xgb_train.py" ]
