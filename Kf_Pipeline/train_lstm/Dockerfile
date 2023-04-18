FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

RUN pip install -U numpy pandas sklearn matplotlib tqdm

COPY train_lstm.py ./train_lstm.py

ENTRYPOINT [ "python", "train_lstm.py" ]
