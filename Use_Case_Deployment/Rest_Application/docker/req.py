import pandas as pd
import numpy as np
import requests, json

data_df = pd.read_csv('../data_for_prediction/engine2.csv')
data_df = data_df.to_json(orient='split')

response = requests.request("POST" , "http://172.30.214.90:5000/api/v0/predict_rul", data=data_df)

RUL = json.loads(response.text)[0]['RUL']

print(RUL)

