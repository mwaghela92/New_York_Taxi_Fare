#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:11:49 2018

@author: mayur 

Added peak hours, night and working day flag - 9.07
"""

import pandas as pd
import datetime
from math import sin, cos, sqrt, atan2
import math
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
import numpy as np
import time

Nrows = 1000000
#Nrows = 1000

#read training data
read_time = time.time()
print('\n\n\nReading data....')
df = pd.read_csv('/Users/mayur/Documents/GitHub/New_York_Taxi_Fare/Data/train.csv',
                 nrows = Nrows).dropna()
end_time = time.time()
print ('Total time to read data: ' + str(round(end_time - read_time, 2))+' s')

#df.columns
#df.describe()

print('\n\n\nPreprocessing started.....')
prep_start = time.time()

#removing rows with fare less than minimum fare valur for ride i.e. 2.5
df = df[df['fare_amount'] > 2.5]

df_test = pd.read_csv('/Users/mayur/Documents/GitHub/New_York_Taxi_Fare/Data/test.csv')
x = df_test.describe()
y = df.describe()

# removing lat and long which are out of bounds as compared ito test set:
min_lat = 40.5
max_lat = 41.8

min_lon = -74.3
max_lon = -72.9

df = df[(df['pickup_latitude'] > min_lat) & 
        (df['pickup_latitude'] < max_lat) &
        (df['dropoff_latitude'] > min_lat) &
        (df['dropoff_latitude'] < max_lat)]
df = df.reset_index(drop = True)


def passenger_flag(df):
    #addin flag for passenger count. If no of passsengers <=4, set as 1 else set as 2
    d = pd.DataFrame(df['passenger_count'])
    d = d.apply(lambda x : pd.cut(x,[-1,4,8],labels=[0,1]))
    df['passenger_cat'] = d.loc[:,'passenger_count']
    return df

def day_encoding(date):
    date1 = date.replace('-', ',')
    date2 = date1.replace(':', ',')
    date3 = date2[:10]
    year,month,date = [int(x) for x in date3.split(',')]
    x = datetime.datetime(year, month,date)
    day = x.weekday()
    return day

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

def dataframe_construction(df):
    dist = list()
    day = list()
    date = list()
    month = list()
    time= list()
    weekday = list()
    year = list()
    workingday = list()
    peak_hours = list()
    night = list()
    
    for i in range(len(df)):
        #i=1
        dt = df.loc[i,'pickup_datetime']
        weekday.append(day_encoding(dt))
        time.append(int(dt[-12:-10]))
        date.append(int(dt[8:10]))
        month.append(int(dt[5:7]))
        year.append(int(dt[0:4]))
        #i=1
        dist.append(distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             df.loc[i,'dropoff_latitude'],df.loc[i,'dropoff_longitude']))
        
        if day_encoding(dt) <6:
            workingday.append(1)
        else:
            workingday.append(0)
        
        if int(dt[-12:-10]) > 15 & int(dt[-12:-10]) < 21:
            peak_hours.append(1)
        else:
            peak_hours.append(0)
        
        if int(dt[-12:-10]) <7:
            night.append(1)
        elif int(dt[-12:-10]) >19 & int(dt[-12:-10]) <25:
            night.append(1)
        else:
            night.append(0)
        """
        if (int(dt[-12:-10]) >7) & (int(dt[-12:-10]) <19):
            peak_hours.append(1)
        else:
            peak_hours.append(0)
        """
    
    df['weekday'] = weekday
    df['distance'] = dist
    df['time'] = time
    df['date'] = date
    df['month']=month
    df['year'] = year
    df['peak_hours'] = peak_hours
    return df
        
 
df.columns

LAT_JFK = 40.6441666
LON_JFK = -73.7822222
LAT_LAGU = 40.7747222
LON_LAGU = -73.8719444
LAT_NEW = 40.6897222
LON_NEW = -74.175

def JFK_related(df):
    dist_pickup_jfk = list()
    dist_dropoff_jfk = list()
    dist_pickup_jfk1 = list()
    dist_dropoff_jfk1 = list()
    dist_pickup_jfk2 = list()
    dist_dropoff_jfk2 = list()
    
    jfk_pickup = list()
    jfk_drop = list()
    lg_pickup = list()
    lg_drop = list()
    nw_pickup = list()
    nw_drop= list()
    
    for i in range(len(df)):
        x = distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             LAT_JFK, LON_JFK)
        y = distance(LAT_JFK, LON_JFK,df.loc[i,'dropoff_latitude'],
                     df.loc[i,'dropoff_longitude'] )
        if x <1:
            dist_pickup_jfk.append(1)
        else: 
            dist_pickup_jfk.append(0)
        jfk_pickup.append(x)
        
        if y<1:
            dist_dropoff_jfk.append(1)
        else:
            dist_dropoff_jfk.append(0)
        jfk_drop.append(y)
            
            
        x1 = distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             LAT_LAGU, LON_LAGU)
        y1 = distance(LAT_LAGU, LON_LAGU,df.loc[i,'dropoff_latitude'],
                     df.loc[i,'dropoff_longitude'] )
        if x1 <1:
            dist_pickup_jfk1.append(1)
        else: 
            dist_pickup_jfk1.append(0)
        lg_pickup.append(x1)
        if y1<1:
            dist_dropoff_jfk1.append(1)
        else:
            dist_dropoff_jfk1.append(0)
        lg_drop.append(y1)
        
        x2 = distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             LAT_NEW, LON_NEW)
        y2 = distance(LAT_NEW, LON_NEW,df.loc[i,'dropoff_latitude'],
                     df.loc[i,'dropoff_longitude'] )
        if x2 <1:
            dist_pickup_jfk2.append(1)
        else: 
            dist_pickup_jfk2.append(0)
        nw_pickup.append(x2)
        if y2<1:
            dist_dropoff_jfk2.append(1)
        else:
            dist_dropoff_jfk2.append(0)
        nw_drop.append(y2)

    df['pickup_JFK1'] = jfk_pickup
    df['dropoff_JFK1'] = jfk_drop
    df['pickup_LAG1'] = lg_pickup
    df['dropoff_LAG1'] = lg_drop
    df['pickup_NEW1'] = nw_pickup
    df['dropoff_NEW1'] = lg_drop
    
    return df



df = JFK_related(df)
df= dataframe_construction(df)
df = passenger_flag(df)
df = df.drop(df[(df['distance'] <1) & (df['fare_amount']>12)].index)
df = df.drop(df[(df['distance'] < 0.1) & (df['distance']>100)].index)
df = df.reset_index(drop = True)

df_test = JFK_related(df_test)
df_test = dataframe_construction(df_test)
df_test = passenger_flag(df_test)


lar = df.nlargest(50, 'distance')
lar_test = df_test.nlargest(50, 'distance')


df.columns


df_train = df[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'pickup_JFK1', 'dropoff_JFK1', 'pickup_LAG1',
       'dropoff_LAG1', 'pickup_NEW1', 'dropoff_NEW1','weekday', 'distance', 'time', 
       'date', 'month','year', 'passenger_cat', 'peak_hours']]
key = df_test['key']
df_test = df_test[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'pickup_JFK1', 'dropoff_JFK1', 'pickup_LAG1',
       'dropoff_LAG1', 'pickup_NEW1', 'dropoff_NEW1','weekday', 'distance', 'time', 
       'date', 'month','year', 'passenger_cat', 'peak_hours']]

target= df['fare_amount']

prep_end = time.time()
print ('Total time to preprocess data: ' + str(round(prep_start - prep_end, 2))+' s')


df_train.columns == df_test.columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
train = scaler.fit_transform(df_train)
test= scaler.transform(df_test)


from sklearn import metrics, ensemble, linear_model

gb = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=5,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=20, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=1000,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)

#gb.fit(train, target)

print('Training.....')
train_start = time.time()

import xgboost as xgb

params = {
    # Parameters that we are going to tune.
    'max_depth': 8, #Result of tuning with CV
    'eta':.03, #Result of tuning with CV
    'subsample': 1, #Result of tuning with CV
    'colsample_bytree': 0.8, #Result of tuning with CV
    # Other parameters
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 1
}
# params by Jesús Ros XGBoost'ing Taxi Fares, Kaggle

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,target, test_size=0.25)



def XGBoost(X_train,X_test,y_train,y_test):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse',
                             'max_depth': 10, 
                             'eta':.03, 
                                'subsample': 1, 
                                'colsample_bytree': 0.8,
                                'silent': 1
                             }
                    ,dtrain=dtrain,num_boost_round=800, 
                    early_stopping_rounds=75,evals=[(dtest,'test')],)
    
xgbm = XGBoost(X_train,X_test,y_train,y_test)

train_end = time.time()
print ('Total time to train: ' + str(round(train_end - train_start, 2))+' s')

XGBPredictions = xgbm.predict(xgb.DMatrix(test), ntree_limit = xgbm.best_ntree_limit)
y_pred = np.round(XGBPredictions, decimals=2)



result = pd.DataFrame(columns = ['key', 'fare_amount'])
result['key'] = key
result['fare_amount'] = y_pred 

result.to_csv('/Users/mayur/Documents/GitHub/New_York_Taxi_Fare/processed_files/result_19.csv', 
              index= False)


