#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:11:49 2018

@author: mayur
"""

import pandas as pd
import datetime
from math import sin, cos, sqrt, atan2
import math
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
import numpy as np
import time


#read training data
print('Reading data')
start_read = time.time()
df = pd.read_csv('/Users/mayur/Documents/GitHub/New_York_Taxi_Fare/Data/train.csv',
                 nrows = 700000).dropna()
end_read = time.time()
tot_read =end_read - start_read
print('Total time to read = ' + str(tot_read))

df.columns
df.describe()

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
    for i in range(len(df)):
        dt = df.loc[i,'pickup_datetime']
        weekday.append(day_encoding(dt))
        time.append(dt[-12:-10])
        date.append(dt[8:10])
        month.append(dt[5:7])
        year.append(dt[0:4])
        #i=1
        dist.append(distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             df.loc[i,'dropoff_latitude'],df.loc[i,'dropoff_longitude']))
    
    df['weekday'] = weekday
    df['distance'] = dist
    df['time'] = time
    df['date'] = date
    df['month']=month
    df['year'] = year
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
    
    for i in range(len(df)):
        x = distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             LAT_JFK, LON_JFK)
        y = distance(LAT_JFK, LON_JFK,df.loc[i,'dropoff_latitude'],
                     df.loc[i,'dropoff_longitude'] )
        if x <1:
            dist_pickup_jfk.append(1)
        else: 
            dist_pickup_jfk.append(0)
        if y<1:
            dist_dropoff_jfk.append(1)
        else:
            dist_dropoff_jfk.append(0)
            
            
        x1 = distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             LAT_LAGU, LON_LAGU)
        y1 = distance(LAT_LAGU, LON_LAGU,df.loc[i,'dropoff_latitude'],
                     df.loc[i,'dropoff_longitude'] )
        if x1 <1:
            dist_pickup_jfk1.append(1)
        else: 
            dist_pickup_jfk1.append(0)
        if y1<1:
            dist_dropoff_jfk1.append(1)
        else:
            dist_dropoff_jfk1.append(0)
            
        
        x2 = distance(df.loc[i,'pickup_latitude'],df.loc[i,'pickup_longitude'],
                             LAT_NEW, LON_NEW)
        y2 = distance(LAT_NEW, LON_NEW,df.loc[i,'dropoff_latitude'],
                     df.loc[i,'dropoff_longitude'] )
        if x2 <1:
            dist_pickup_jfk2.append(1)
        else: 
            dist_pickup_jfk2.append(0)
        if y2<1:
            dist_dropoff_jfk2.append(1)
        else:
            dist_dropoff_jfk2.append(0)
        
            
            
    #df1 = pd.DataFrame()       
    df['pickup_JFK'] = dist_pickup_jfk
    df['dropoff_JFK'] = dist_dropoff_jfk
    df['pickup_LAG'] = dist_pickup_jfk1
    df['dropoff_LAG'] = dist_dropoff_jfk1
    df['pickup_NEW'] = dist_pickup_jfk2
    df['dropoff_NEW'] = dist_dropoff_jfk2
    
    return df

start_pp = time.time()

df = JFK_related(df)
df= dataframe_construction(df)
df = passenger_flag(df)
df = df.drop(df[(df['distance'] <1) & (df['fare_amount']>12)].index)
df = df.drop(df[(df['distance'] < 0.1) & (df['distance']>100)].index)
df = df.reset_index(drop = True)

df_test1 = JFK_related(df_test)
df_test1 = dataframe_construction(df_test)
df_test1 = passenger_flag(df_test)


key = df_test['key']
target= df['fare_amount']

end_pp = time.time()
tot_pp = end_pp - start_pp
print('total pre processing time' + str(tot_pp))

lar = df.nlargest(50, 'distance')
lar_test = df_test.nlargest(50, 'distance')


df.columns



df_train = df[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'weekday', 'distance', 'time', 
       'date', 'month','year', 'passenger_cat']]

df_test = df_test1[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count','weekday', 'distance', 'time', 
       'date', 'month','year', 'passenger_cat']]
"""
df_train = df[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'pickup_JFK', 'dropoff_JFK', 'pickup_LAG',
       'dropoff_LAG', 'pickup_NEW', 'dropoff_NEW','weekday', 'distance', 'time', 
       'date', 'month','year', 'passenger_cat']]

df_test = df_test[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'pickup_JFK', 'dropoff_JFK', 'pickup_LAG',
       'dropoff_LAG', 'pickup_NEW', 'dropoff_NEW','weekday', 'distance', 'time', 
       'date', 'month','year', 'passenger_cat']]
"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
train = scaler.fit_transform(df_train)
test= scaler.transform(df_test)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

df.columns
train = pd.concat([train, df[['pickup_JFK', 'dropoff_JFK', 'pickup_LAG',
       'dropoff_LAG', 'pickup_NEW', 'dropoff_NEW']]], axis = 1)
test = pd.concat([test, df_test1[['pickup_JFK', 'dropoff_JFK', 'pickup_LAG',
       'dropoff_LAG', 'pickup_NEW', 'dropoff_NEW']]], axis = 1)


df_train.columns == df_test.columns


from sklearn import metrics, ensemble, linear_model

start_fit = time.time()

gb = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=3,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=15, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=1000,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)

gb.fit(train, target)

end_fit = time.time()
tot_fit = end_fit - start_fit
print ('Total fit time = ' + str(tot_fit))

y_pred = gb.predict(test)

result = pd.DataFrame(columns = ['key', 'fare_amount'])
result['key'] = key
result['fare_amount'] = y_pred 

result.to_csv('/Users/mayur/Documents/GitHub/New_York_Taxi_Fare/processed_files/result_11.csv', 
              index= False)



