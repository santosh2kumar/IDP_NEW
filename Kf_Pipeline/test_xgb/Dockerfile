FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas sklearn xgboost

COPY test_xgb.py ./test_xgb.py

ENTRYPOINT [ "python", "test_xgb.py" ]
