FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

RUN pip install -U numpy pandas sklearn matplotlib tqdm

COPY test_lstm.py ./test_lstm.py

ENTRYPOINT [ "python", "test_lstm.py" ]
