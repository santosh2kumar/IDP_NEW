FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas tensorflow keras sklearn flask

COPY app.py ./app.py
COPY model-00084-928.00182-0.77142-0.73563.h5 ./model-00084-928.00182-0.77142-0.73563.h5
COPY min_max_scaler.pkl ./min_max_scaler.pkl

ENTRYPOINT [ "python", "app.py" ]

