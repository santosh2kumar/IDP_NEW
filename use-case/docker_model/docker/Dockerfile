FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas sklearn flask requests

COPY api.py ./api.py
COPY model.pkl ./model.pkl

ENTRYPOINT [ "python", "api.py" ]

