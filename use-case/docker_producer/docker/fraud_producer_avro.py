from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
import pandas as pd
import glob
from time import sleep
value_schema = avro.load('/mnt/value-schema-fraud.avsc')
key_schema = avro.load('/mnt/key-schema-fraud.avsc')

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))


avroProducer = AvroProducer({
    'bootstrap.servers': 'kafka-cp-kafka:9092',
    'on_delivery': delivery_report,
    'schema.registry.url': 'http://cp-schema-registry:8081'
    }, default_key_schema=key_schema, default_value_schema=value_schema)

transaction_cnt = 0

while(True):
    files = glob.glob("/mnt/test*")
    for file in files:
        data_df = pd.read_csv(file)
        data_df = data_df.to_dict(orient='records')
        #file_name = file.split("/")[2].split(".")[0]
        #data = {file_name : data_df}
        for data in data_df:
            print(data)
            transaction_cnt += 1
            key = {'transaction_cnt': transaction_cnt}
            value = data
            try:
                avroProducer.produce(topic='fraud_avro1', value=value, key=key)
                avroProducer.flush()
            except Exception as e:
                print('Error producing msg...')
                print(e)
            if(transaction_cnt % 10 == 0):
                sleep(0.25)
    while(True):
        sleep(5)
