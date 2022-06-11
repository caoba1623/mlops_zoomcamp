import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val     = dv.transform(val_dicts) 
    y_pred    = lr.predict(X_val)
    y_val     = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")

def month_format(month):
    if month < 10:
        return "0" + str(month)
    else:
        return str(month)

@task
def get_paths(date:str):
    date_time_obj = None
    
    if date:
        date_time_obj = datetime. strptime(date, '%Y-%m-%d')
            
    else:
        date_time_obj = datetime.now()
        
    train_date    = date_time_obj + relativedelta(months=-2)
    val_date      = date_time_obj + relativedelta(months=-1)
    
    train_month = train_date.month
    train_year  = train_date.year
    val_month   = val_date.month
    val_year    = val_date.year
    
    str_train_date = str(train_year) + "-" + month_format(train_month)
    str_val_date   = str(val_year)   + "-" + month_format(val_month)
    
    return f"fhv_tripdata_{str_train_date}.parquet", f"fhv_tripdata_{str_val_date}.parquet"

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date).result()

    df_train = read_data("./data/" + train_path)
    df_train_processed = prepare_features(df_train, categorical)#.result()

    df_val = read_data("./data/" + val_path)
    df_val_processed = prepare_features(df_val, categorical, False)#.result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # Save models
    with open(f"./models/model-{date}.bin", 'wb') as f_out:
        pickle.dump(lr, f_out)

    with open(f"./models/dv-{date}.bin", 'wb') as f_out:
        pickle.dump(dv, f_out)

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name='model_training_homework_03',
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml_prefect_run'],
)

#main(date="2021-08-15")