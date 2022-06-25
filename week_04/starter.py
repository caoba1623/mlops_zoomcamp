import pickle
import pandas as pd
import numpy as np
import argparse

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    categorical = ['PUlocationID', 'DOlocationID']

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet', categorical)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(np.mean(y_pred))
    
    year  = int(year)
    month = int(month)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df['predictions'] = y_pred

    df.to_parquet(
        path="predict.parquet",
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    parser    = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
 
    params = subparser.add_parser('params')
    params.add_argument('--year', type=str, required=True)
    params.add_argument('--month', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'params':

        year  = args.year
        month = args.month

        main(year, month)