#from kafka import KafkaConsumer
from confluent_kafka import Consumer
from cassandra.cluster import Cluster
from json import loads
import requests
import pandas as pd
from datetime import date
from mapr.ojai.storage.ConnectionFactory import ConnectionFactory

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

consumer.subscribe(['nasa_predict'])

cluster = Cluster(['cassandra'], port=9042, control_connection_timeout=None)
session = cluster.connect()
session.execute("USE nasa")

connection_str = "172.30.203.1:5678?auth=basic;user=mapr;password=mapr;ssl=true;sslCA=/app/ssl_truststore.pem;sslTargetNameOverride=dfdn01.ecp.local"

connection = ConnectionFactory.get_connection(connection_str=connection_str)

try:
    store = connection.get_store('/prediction_data')
except:
    store = connection.create_store('/prediction_data')

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

    data_values = dataset.values.tolist()
    today = date.today()
    recorded_date = today.strftime("%Y-%m-%d")

    mapr_data = {'engine' : engine, '_id' : engine+'_'+recorded_date , 'data' : data_values}

    new_document = connection.new_document(dictionary=mapr_data)
    store.insert_or_replace(new_document)

    previous_day_data = dataset.iloc[-1]
    engine_details = {}
    
    time_in_cycles = previous_day_data[0]
    setting_1 = previous_day_data[1]
    setting_2 = previous_day_data[2]
    T2 = previous_day_data[3]
    T24 = previous_day_data[4]
    T30 = previous_day_data[5]
    T50 = previous_day_data[6]
    Nf = previous_day_data[7]
    Nc = previous_day_data[8]
    epr = previous_day_data[9]
    Ps30 = previous_day_data[10]
    NRf = previous_day_data[11]
    NRc = previous_day_data[12]
    BPR = previous_day_data[13]
    farB = previous_day_data[14]
    htBleed = previous_day_data[15]
    Nf_dmd = previous_day_data[16]
    W31 = previous_day_data[17]
    W32 = previous_day_data[18]

    today = date.today()
    recorded_date = today.strftime("%Y-%m-%d")
  
    query_string = "INSERT INTO engine_details (engine, recorded_date, RUL, time_in_cycles, setting_1, setting_2, T2, T24, T30, T50, Nf, Nc, epr, Ps30, NRf, NRc, BPR, farB, htBleed, Nf_dmd, W31, W32) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(engine, recorded_date, RUL, time_in_cycles, setting_1, setting_2, T2, T24, T30, T50, Nf, Nc, epr, Ps30, NRf, NRc, BPR, farB, htBleed, Nf_dmd, W31, W32)

    print(query_string)
    session.execute(query_string)
