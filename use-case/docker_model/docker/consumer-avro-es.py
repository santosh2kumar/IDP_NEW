from confluent_kafka.avro import AvroConsumer
from confluent_kafka.avro.serializer import SerializerError
import pandas as pd
import requests
from json import loads
from datetime import date
from elasticsearch import Elasticsearch
from mapr.ojai.storage.ConnectionFactory import ConnectionFactory

c = AvroConsumer({
    'bootstrap.servers': 'kafka-cp-kafka:9092',
    'group.id': 'groupid',
    'schema.registry.url': 'http://cp-schema-registry:8081'})

c.subscribe(['nasa_avro_es'])

es = Elasticsearch("http://elasticsearch.default.svc:9200")

connection_str = "172.30.203.1:5678?auth=basic;user=mapr;password=mapr;ssl=true;sslCA=/mnt/ssl_truststore.pem;sslTargetNameOverride=dfdn01.ecp.local"

connection = ConnectionFactory.get_connection(connection_str=connection_str)

try:
    store = connection.get_store('/prediction_data')
except:
    store = connection.create_store('/prediction_data')

count = 0
data = []

while True:
    try:
        msg = c.poll(1)

    except SerializerError as e:
        print("Message deserialization failed for {}: {}".format(msg, e))
        break

    if msg is None:
        continue

    if msg.error():
        print("AvroConsumer error: {}".format(msg.error()))
        continue

    data.append(msg.value())
    count += 1
    if(count == 100):
        data_df = pd.DataFrame(data)
        engine = msg.key()['name']

        message = data_df.to_json(orient='split')

        response = requests.request("POST" , "http://model-endpoint:5000/api/v0/predict_rul", data=message)
        RUL = loads(response.text)[0]['RUL']

        today = date.today()
        recorded_date = today.strftime("%Y-%m-%d")
        mapr_data = {'engine' : engine, '_id' : engine+'_'+recorded_date, 'date' : recorded_date, 'data' : data}

        new_document = connection.new_document(dictionary=mapr_data)
        store.insert_or_replace(new_document)

        print(engine)
        print(RUL)

        previous_day_data = data[-1]
        previous_day_data['engine'] = engine
        previous_day_data['RUL'] = RUL
        today = date.today()
        previous_day_data['recorded_date'] = today
        recorded_date = today.strftime("%Y-%m-%d")

        res = es.index(index="engine_details", id=engine+'_'+recorded_date, document=previous_day_data)

        data = []
        count = 0

c.close()
