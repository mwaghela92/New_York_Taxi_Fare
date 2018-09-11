#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:14:20 2018

@author: mayur
"""


import pandas as pd
from nltk.tokenize import word_tokenize 
import datetime
from math import sin, cos, sqrt, atan2
import math
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

df2 = pd.read_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/Data/train.csv',
                 nrows = 1000000).dropna()
df = df2
df.head
df_col = list(df.columns)

def day_encoding(date):
    date1 = date.replace('-', ',')
    date2 = date1.replace(':', ',')
    date3 = date2[:10]
    year,month,date = [int(x) for x in date3.split(',')]
    x = datetime.datetime(year, month,date)
    day = x.weekday()
    return day

    
def get_distance(lat_1, lng_1, lat_2, lng_2): 
    d_lat = lat_2 - lat_1
    #print(d_lat)
    d_lng = lng_2 - lng_1 
    #print(d_lng)

    temp = (  
         math.sin(d_lat / 2) ** 2 
       + math.cos(lat_1) 
       * math.cos(lat_2) 
       * math.sin(d_lng / 2) ** 2
    )

    return 6.3730 * (2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp)))



y = list()
distance = list()
time_hour = list()
date1 = list()
month = list()

lon1 = df['pickup_longitude']
lon2 = df['dropoff_longitude']
lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']


for i in range(len(df)):
    if i == 120227 or i == 245696 or i == 340533 or i == 428108 or i == 471472 or i == 524834 or i == 574023 or i == 580338 or i == 794694 or i == 895400 :
        y.append(0)
        time_hour.append(1)
        distance.append(0)
        date1.append(1)
        month.append(1)
    else:  
        date = df.iloc[i,2]
        y.append(day_encoding(date))
        time_hour.append(date[-12:-10])
        date1.append(date[8:10])
        month.append(date[5:7])
        
        distance.append(get_distance(lat1[i], lat2[i], lon1[i], lon2[i]))

remove_index = [120227,245696,340533,428108,471472,524834,
                574023 ,580338 ,794694 ,895400]     

    
    


df['weekday'] = y

#from sklearn import preprocessing
#distance = preprocessing.scale(distance)

df['distance'] = distance
df['time_hour'] = time_hour
df['date1'] = date1
df['month']=month
print(set(time_hour))
df = df.drop(df.index[remove_index])

cols = ['pickup_longitude','pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'weekday', 'distance', 
        'time_hour', 'date1', 'month']

train_df = df[cols] 

import numpy as np
inds = np.asarray(train_df.isnull()).nonzero()
train_df.dropna(inplace = True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # create an instance
train = scaler.fit_transform(train_df)

train = pd.DataFrame(train)

if len(inds[0])>0:
    df = df.drop(df.index[[inds[0]]])


d = pd.DataFrame(df['passenger_count'])
d = d.apply(lambda x : pd.cut(x,[-1,4,8],labels=[0,1]))
train['passenger_cat'] = d.loc[:,'passenger_count']

inds1 = np.asarray(train.isnull()).nonzero()
if len(inds1[0])>0:
    train = train.drop(train.index[[inds1[0]]])
if len(inds1[0])>0:   
    df = df.drop(df.index[[inds1[0]]])
target = df['fare_amount']


from sklearn import metrics, ensemble, linear_model

gb = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=3,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=15, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=1000,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)

gb.fit(train, target)





df2 = pd.read_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/Data/test.csv').dropna()
df = df2
df.head
df_col = list(df.columns)


y = list()
distance = list()
time_hour = list()
date1 = list()
month = list()

lon1 = df['pickup_longitude']
lon2 = df['dropoff_longitude']
lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']


for i in range(len(df)):
    if i == 120227 or i == 245696 or i == 340533 or i == 428108 or i == 471472 or i == 524834 or i == 574023 or i == 580338 or i == 794694 or i == 895400 :
        y.append(0)
        time_hour.append(1)
        distance.append(0)
        date1.append(1)
        month.append(1)
    else:  
        date = df.iloc[i,1]
        y.append(day_encoding(date))
        time_hour.append(date[-12:-10])
        date1.append(date[8:10])
        month.append(date[5:7])
        
        distance.append(get_distance(lat1[i], lat2[i], lon1[i], lon2[i]))
       
df['weekday'] = y

#from sklearn import preprocessing
#distance = preprocessing.scale(distance)

df['distance'] = distance
df['time_hour'] = time_hour
df['date1'] = date1
df['month']=month
print(set(time_hour))



cols = ['pickup_longitude','pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'weekday', 'distance', 
        'time_hour', 'date1', 'month']

test_df = df[cols]
 # create an instance
test_df = scaler.transform(test_df) 
test_df = pd.DataFrame(test_df)

d = pd.DataFrame(df['passenger_count'])
d = d.apply(lambda x : pd.cut(x,[-1,4,8],labels=[0,1]))
test_df['passenger_cat'] = d['passenger_count'] 

test_df.columns == train.columns
y_pred = gb.predict(test_df)

result = pd.DataFrame(columns = ['key', 'fare_amount'])
result['key'] = df2['key']
result['fare_amount'] = y_pred 

result.to_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/processed_files/result_08.csv', 
              index= False)









