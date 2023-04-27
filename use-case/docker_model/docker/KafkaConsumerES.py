#from kafka import KafkaConsumer
from confluent_kafka import Consumer
#from cassandra.cluster import Cluster
from elasticsearch import Elasticsearch
from json import loads
import requests
import pandas as pd
from datetime import date

#consumer = KafkaConsumer(
#    'nasa_predict',
#     bootstrap_servers=['kafka-svc:9093'],
#     auto_offset_reset='earliest',
#     enable_auto_commit=True,
#     group_id='my-group',
#     value_deserializer=lambda x: loads(x.decode('utf-8')))

consumer = Consumer({
    'bootstrap.servers': 'kafka-cp-kafka:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True
})

consumer.subscribe(['nasa_predict_es'])

#cluster = Cluster(['cassandra'], port=9042, control_connection_timeout=None)
#session = cluster.connect()
#session.execute("USE nasa")

es = Elasticsearch("http://elasticsearch.default.svc:9200")

while(True):
    message = consumer.poll(0.1)
    if message is None:
        continue
    if message.error():
        print("Consumer error: {}".format(message.error()))
        continue
    message = loads(message.value().decode('utf-8'))
    engine = list(message.keys())[0]
    message = message[engine]
   #print(engine)
    
    response = requests.request("POST" , "http://model-endpoint:5000/api/v0/predict_rul", data=message)
    RUL = loads(response.text)[0]['RUL']
    dataset = pd.read_json(message, typ='frame', orient='split')
    previous_day_data = dataset.iloc[-1]
    engine_details = {}
    es_data = {}

    es_data['engine'] = engine
    es_data['RUL'] = RUL
    es_data['time_in_cycles'] = previous_day_data[0]
    es_data['setting_1'] = previous_day_data[1]
    es_data['setting_2'] = previous_day_data[2]
    es_data['T2'] = previous_day_data[3]
    es_data['T24'] = previous_day_data[4]
    es_data['T30'] = previous_day_data[5]
    es_data['T50'] = previous_day_data[6]
    es_data['Nf'] = previous_day_data[7]
    es_data['Nc'] = previous_day_data[8]
    es_data['epr'] = previous_day_data[9]
    es_data['Ps30'] = previous_day_data[10]
    es_data['NRf'] = previous_day_data[11]
    es_data['NRc'] = previous_day_data[12]
    es_data['BPR'] = previous_day_data[13]
    es_data['farB'] = previous_day_data[14]
    es_data['htBleed'] = previous_day_data[15]
    es_data['Nf_dmd'] = previous_day_data[16]
    es_data['W31'] = previous_day_data[17]
    es_data['W32'] = previous_day_data[18]

    today = date.today()
    es_data['recorded_date'] = today
    recorded_date = today.strftime("%Y-%m-%d")

    res = es.index(index="engine_details", id=engine+'_'+recorded_date, document=es_data)
    print(res['result'])
