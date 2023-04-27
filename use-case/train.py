import sys

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import mlflow
import mlflow.sklearn

def preprocess_data(df):
    df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

    df['step_day'] = df['step'].map(lambda x: x//24)
    df['hour'] = df['step'].map(lambda x: x%24)
    df['step_week'] = df['step_day'].map(lambda x: x//7)

    df['from_to'] = df['nameOrig'].str[0] + df['nameDest'].str[0]

    df_get_type = pd.get_dummies(df['type'], drop_first=True)
    df_from_to = pd.get_dummies(df['from_to'], drop_first=True)

    df = pd.concat([df,df_get_type],axis=1)
    df = pd.concat([df,df_from_to],axis=1)

    df['errorOrig'] = df['amount'] + df['newBalanceOrig'] - df['oldBalanceOrig']
    df['errorDest'] = df['amount'] + df['oldBalanceDest'] - df['newBalanceDest']

    df_clean = df.drop(columns=['type','from_to','nameOrig','nameDest','isFlaggedFraud','step'], axis=1)

    X = df_clean.drop(['isFraud'], axis=1)
    y = df_clean['isFraud']

    return(X, y)

def evaluate(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    p_train = metrics.precision_score(y_train, y_pred_train)
    r_train = metrics.recall_score(y_train, y_pred_train)
    print("Train Precesion: {}. Train recall: {}".format(p_train, r_train))
    
    y_pred_test = model.predict(X_test)
    p_test = metrics.precision_score(y_test, y_pred_test)
    r_test = metrics.recall_score(y_test, y_pred_test)
    
    return(p_train, r_train, p_test, r_test)

def train_dt(params):
    mlflow.sklearn.autolog()
    with mlflow.start_run(nested=True):
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_train, y_train)

        predictions_test = dt.predict(X_test)
        auc_score = metrics.roc_auc_score(y_test, predictions_test)

        mlflow.log_metric('auc', auc_score)
        mlflow.sklearn.log_model(dt, "model")

        return {'status': STATUS_OK, 'loss': -1*auc_score, 'dt': dt.get_params(deep=True)}

def train_rf(params):
    mlflow.sklearn.autolog()
    with mlflow.start_run(nested=True):
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        predictions_test = rf.predict(X_test)
        auc_score = metrics.roc_auc_score(y_test, predictions_test)

        mlflow.log_metric('auc', auc_score)
        mlflow.sklearn.log_model(rf, "model")

        return {'status': STATUS_OK, 'loss': -1*auc_score, 'rf': rf.get_params(deep=True)}


df = pd.read_csv('/mnt/data/PS_20174392719_1491204439457_log.csv')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'dt'
    max_evals = int(sys.argv[2]) if len(sys.argv) > 1 else 10    

    if(model == 'dt'):
        search_space = {
        "class_weight": hp.choice('class_weight', [{0:0.1, 1:0.9}, {0:1, 1:1}]),
        "max_depth": hp.choice('max_depth', [2, 5, 10, 15]),
        "min_samples_leaf": hp.choice('min_samples_leaf', [10, 20, 50]),
        "max_features": hp.choice('max_features', [5, 10, 12])
        }

        mlflow.set_tracking_uri("postgresql://mlflow_user:mlflow_pwd@mlflow-postgres-service:5432/mlflow_db")
        try:
            mlflow.create_experiment('Decision Trees Training Fraud Detection', '/mnt/artifacts/dt')
        except:
            print("Experiment already present. Continuing")
        mlflow.set_experiment("Decision Trees Training Fraud Detection")
        with mlflow.start_run(run_name='Decision_Tree_models'):
            best_params = fmin(
              fn=train_dt, 
              space=search_space, 
              algo=tpe.suggest, 
              max_evals=max_evals
            )
    elif(model == 'rf'):
        search_space = {
        "class_weight": hp.choice('class_weight', [{0:0.1, 1:0.9}, {0:1, 1:1}]),
        "max_depth": hp.choice('max_depth', [3, 5, 10]),
        "min_samples_leaf": hp.choice('min_samples_leaf', [10, 20, 50, 100]),
        "max_features": hp.choice('max_features', [3, 5, 10]),
        "n_estimators": hp.choice('n_estimators', [10, 25, 50])
        }

        mlflow.set_tracking_uri("postgresql://mlflow_user:mlflow_pwd@mlflow-postgres-service:5432/mlflow_db")
        try:
            mlflow.create_experiment('Random Forest Training Fraud Detection', '/mnt/artifacts/rf')
        except:
            print("Experiment already present. Continuing")
        mlflow.set_experiment("Random Forest Training Fraud Detection")
        with mlflow.start_run(run_name='Random_Forest_models'):
            best_params = fmin(
              fn=train_rf,
              space=search_space,
              algo=tpe.suggest,
              max_evals=max_evals
            )

if(__name__ == '__main__'):
    main()
