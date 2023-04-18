import pandas as pd
from json import dumps
from kafka import KafkaProducer
from time import sleep
import glob

producer = KafkaProducer(bootstrap_servers=['kafka-svc:9093'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))

while(True):
    files = glob.glob("/mnt/*")
#    for file in files:
#        data_df = pd.read_csv(file)
#        data_df = data_df.to_json(orient='split')
#        file_name = file.split("/")[2].split(".")[0]
#        data = {file_name : data_df}
#        print(file_name)
#        future = producer.send('nasa_predict', value=data)
#        result = future.get(timeout=60)
    sleep(5)
#    break

