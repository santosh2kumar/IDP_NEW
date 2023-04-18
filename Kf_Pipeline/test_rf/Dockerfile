FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas sklearn

COPY test_rf.py ./test_rf.py

ENTRYPOINT [ "python", "test_rf.py" ]
