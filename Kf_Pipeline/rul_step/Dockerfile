FROM python:3.8-slim

WORKDIR /app

RUN pip install -U numpy pandas

COPY calculate_rul.py ./calculate_rul.py

ENTRYPOINT [ "python", "calculate_rul.py" ]
