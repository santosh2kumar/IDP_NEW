from kafka import KafkaConsumer
from cassandra.cluster import Cluster
from json import loads
import requests
import pandas as pd
from datetime import date

consumer = KafkaConsumer(
    'nasa_predict',
     bootstrap_servers=['kafka-svc:9093'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group',
     value_deserializer=lambda x: loads(x.decode('utf-8')))

cluster = Cluster(['cassandra'], port=9042, control_connection_timeout=None)
session = cluster.connect()
session.execute("USE nasa")

for message in consumer:
    message = message.value
    engine = list(message.keys())[0]
    message = message[engine]
   #print(engine)
    
    response = requests.request("POST" , "http://model-endpoint:5000/api/v0/predict_rul", data=message)
    RUL = loads(response.text)[0]['RUL']
    dataset = pd.read_json(message, typ='frame', orient='split')
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
