from kafka import KafkaConsumer
from rediscluster import RedisCluster
from json import loads
import requests
import pandas as pd
from datetime import date

consumer = KafkaConsumer(
    'nasa_predict',
     bootstrap_servers=['kafka-svc:9093'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='rul-prediction-group',
     value_deserializer=lambda x: loads(x.decode('utf-8')))

startup_nodes = [{"host": "redis-cluster", "port": "6379"}]
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

for message in consumer:
    message = message.value
    engine = list(message.keys())[0]
    message = message[engine]
    #print(engine)
    #print(message)
    
    response = requests.request("POST" , "http://model-endpoint:5000/api/v0/predict_rul", data=message)
    RUL = loads(response.text)[0]['RUL']
    dataset = pd.read_json(message, typ='frame', orient='split')
    previous_day_data = dataset.iloc[-1]
    engine_details = {}
    engine_details['engine'] = engine
    engine_details['RUL'] = RUL
    engine_details['time_in_cycles'] = previous_day_data[0]
    engine_details['setting_1'] = previous_day_data[1]
    engine_details['setting_2'] = previous_day_data[2]
    engine_details['T2'] = previous_day_data[3]
    engine_details['T24'] = previous_day_data[4]
    engine_details['T30'] = previous_day_data[5]
    engine_details['T50'] = previous_day_data[6]
    engine_details['Nf'] = previous_day_data[7]
    engine_details['Nc'] = previous_day_data[8]
    engine_details['epr'] = previous_day_data[9]
    engine_details['Ps30'] = previous_day_data[10]
    engine_details['NRf'] = previous_day_data[11]
    engine_details['NRc'] = previous_day_data[12]
    engine_details['BPR'] = previous_day_data[13]
    engine_details['farB'] = previous_day_data[14]
    engine_details['htBleed'] = previous_day_data[15]
    engine_details['Nf_dmd'] = previous_day_data[16]
    engine_details['W31'] = previous_day_data[17]
    engine_details['W32'] = previous_day_data[18]

    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    redis_key = engine[:-1] + ":" + engine[-1] + ":" + d1
    rc.hmset(redis_key, engine_details)

    #print(engine_details)
    print(redis_key)
    print(RUL)
