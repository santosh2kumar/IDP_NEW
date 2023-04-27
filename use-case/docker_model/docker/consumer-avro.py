from confluent_kafka.avro import AvroConsumer
from confluent_kafka.avro.serializer import SerializerError
import pandas as pd
import requests
from json import loads
from datetime import date
from cassandra.cluster import Cluster
from mapr.ojai.storage.ConnectionFactory import ConnectionFactory

c = AvroConsumer({
    'bootstrap.servers': 'kafka-cp-kafka:9092',
    'group.id': 'groupid',
    'schema.registry.url': 'http://cp-schema-registry:8081'})

c.subscribe(['nasa_avro'])

cluster = Cluster(['cassandra'], port=9042, control_connection_timeout=None)
session = cluster.connect()
session.execute("USE nasa")

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

        previous_day_data = list(data[-1].values())
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

        data = []
        count = 0

        session.execute(query_string)

c.close()
