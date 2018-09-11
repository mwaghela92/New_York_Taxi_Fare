#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:06:21 2018

@author: mayur
"""

import pandas as pd
from nltk.tokenize import word_tokenize 
import datetime
from math import sin, cos, sqrt, atan2
import math

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

lon1 = df['pickup_longitude']
lon2 = df['dropoff_longitude']
lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']


for i in range(len(df)):
    if i == 120227 or i == 245696 or i == 340533 or i == 428108 or i == 471472 or i == 524834 or i == 574023 or i == 580338 or i == 794694 or i == 895400 :
        y.append(0)
        time_hour.append(1)
        distance.append(0)
    else:  
        date = df.iloc[i,2]
        y.append(day_encoding(date))
        time_hour.append(date[-12:-10])
        distance.append(get_distance(lat1[i], lat2[i], lon1[i], lon2[i]))

remove_index = [120227,245696,340533,428108,471472,524834,
                574023 ,580338 ,794694 ,895400]     

    
    
    
df['weekday'] = y

from sklearn import preprocessing
distance = preprocessing.scale(distance)

df['distance'] = distance
df['time_hour'] = time_hour
print(set(time_hour))

d = pd.DataFrame(df['passenger_count'])
d = d.apply(lambda x : pd.cut(x,[-1,4,8],labels=[0,1]))
df['passenger_cat'] = d['passenger_count']


d2 = pd.get_dummies(df['weekday'], drop_first=True,  prefix= ['weekday'])
df = pd.concat([df,d2], axis = 1)

s = pd.to_numeric(df['time_hour'], errors='coerce')
df['time_hour'] = s
d3 = pd.get_dummies(df['time_hour'], drop_first=True,  prefix= ['time_hour'])
df = pd.concat([df,d3], axis = 1)

type(df['time_hour'])
df2 = df2.drop(df2.index[remove_index])

train_df = df.iloc[:, 9:]
del train_df['time_hour']

train_df['target'] = df['fare_amount']

train_df.isnull().values.any()
len(train_df)
train_df.dropna(inplace = True)

target = train_df['target']
del train_df['target']
#train_df.to_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/processed_files/train_derived.csv')



from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error

gb = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=3,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=15, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=1000,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)

gb.fit(train_df, target)





"""



df = pd.read_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/Data/train.csv',
                 skiprows=1000002, nrows=999)
df.columns = df_col
y = list()
distance = list()
time_hour = list()

lon1 = df['pickup_longitude']
lon2 = df['dropoff_longitude']
lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']


for i in range(len(df)):
    date = df.iloc[i,2]
    y.append(day_encoding(date))
    time_hour.append(date[-12:-10])
    distance.append(get_distance(lat1[i], lat2[i], lon1[i], lon2[i]))
    

    
    
    
df['weekday'] = y
df['distance'] = distance
df['time_hour'] = time_hour


d = pd.DataFrame(df['passenger_count'])
d = d.apply(lambda x : pd.cut(x,[-1,4,8],labels=[0,1]))
df['passenger_cat'] = d['passenger_count']


d2 = pd.get_dummies(df['weekday'], drop_first=True)
df = pd.concat([df,d2], axis = 1)

d3 = pd.get_dummies(df['time_hour'], drop_first=True)
df = pd.concat([df,d3], axis = 1)

test_df = df.iloc[:, 9:]
del test_df['time_hour']


print( all(test_df.columns == train_df.columns))

y_pred = gb.predict(test_df)

y_act = df['fare_amount']

mean_squared_error(y_pred, y_act)

"""


df1 = pd.read_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/Data/test.csv')
df= df1
y = list()
distance = list()
time_hour = list()

lon1 = df['pickup_longitude']
lon2 = df['dropoff_longitude']
lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']


for i in range(len(df)):
    date = df.iloc[i,1]
    y.append(day_encoding(date))
    time_hour.append(date[-12:-10])
    distance.append(get_distance(lat1[i], lat2[i], lon1[i], lon2[i]))
    

    
    
    
df['weekday'] = y

distance = preprocessing.scale(distance)
df['distance'] = distance
df['time_hour'] = time_hour


d = pd.DataFrame(df['passenger_count'])
d = d.apply(lambda x : pd.cut(x,[-1,4,8],labels=[0,1]))
df['passenger_cat'] = d['passenger_count']


d2 = pd.get_dummies(df['weekday'], drop_first=True, prefix= ['weekday'])
df = pd.concat([df,d2], axis = 1)

d3 = pd.get_dummies(df['time_hour'], drop_first=True, prefix= ['time_hour'])
df = pd.concat([df,d3], axis = 1)

test_df = df.iloc[:, 8:]
del test_df['time_hour']
#test_df["""['time_hour']_00"""] = [0] * 9914

rearrange_columns = train_df.columns

#test_df = test_df[rearrange_columns]

#print( all(test_df.columns == train_df.columns))


y_pred = gb.predict(test_df)

#y_act = df['fare_amount']

#mean_squared_error(y_pred, y_act)
train_df.head(2)

result = pd.DataFrame(columns = ['key', 'fare_amount'])
result['key'] = df1['key']
result['fare_amount'] = y_pred 

result.to_csv('/Users/mayur/Documents/GitHub/Taxi_Fare/processed_files/result_03.csv', 
              index= False)





    

