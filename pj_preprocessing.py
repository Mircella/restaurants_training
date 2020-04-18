import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

#geoplaces stats (main)
geo = pd.read_csv('geoplaces.csv', delimiter = ';',encoding='latin-1')
geo.rename(columns={'s': 'placeID'}, inplace = True)

#remaining stats
rating = pd.read_csv('rating_final.csv', delimiter = ';')
rest_payment = pd.read_csv('chefmozaccepts.csv', delimiter = ';')
rest_cuisine = pd.read_csv('chefmozcuisine.csv', delimiter = ';')
rest_hours = pd.read_csv('chefmozhours.csv', delimiter = ',')
rest_parking = pd.read_csv('chefmozparking.csv', delimiter = ';')


def compare_with_geo(dep_unique, dep_table):
    
    dep_unique = dep_table['placeID'].unique
    print(dep_table.__name__) 
    
def get_value(dep_table,x):
    x = dep_table['placeID'].unique()

get_value(rating, x)

compare_with_geo(rating_unique,rating)

#check restaurant ID between geo and rating
rating = pd.read_csv('rating_final.csv', delimiter = ';')

rating_unique = rating['placeID'].unique()


exist = []
for x in rating_unique:
    if x in geo['placeID'].unique():
        exist.append(x)

#check restaurant ID between payment and rating
rest_payment = pd.read_csv('chefmozaccepts.csv', delimiter = ';')
rest_payment_unique = rest_payment['placeID'].unique()

exist_payment = []
for x in rest_payment_unique:
    if x in geo['placeID'].unique():
        exist_payment.append(x)

#check restaurant ID between cuisine and rating
rest_cuisine = pd.read_csv('chefmozcuisine.csv', delimiter = ';')
rest_cuisine_unique = rest_cuisine['placeID'].unique()      
  
exist_cuisine = []
for x in rest_cuisine_unique:
    if x in geo['placeID'].unique():
        exist_cuisine.append(x)   

#check restaurant ID between hours and rating
rest_hours = pd.read_csv('chefmozhours4.csv', delimiter = ';')
rest_hours_unique = rest_hours['placeID'].unique()      
  
exist_hours = []
for x in rest_hours_unique:
    if x in geo['s'].unique():
        exist_hours.append(x) 
        

#check restaurant ID between hours and rating
rest_parking = pd.read_csv('chefmozparking.csv', delimiter = ';')
rest_parking_unique = rest_parking['placeID'].unique()      
  
exist_parking = []
for x in rest_parking_unique:
    if x in geo['s'].unique():
        exist_parking.append(x) 

import functools
new = functools.reduce(functools.partial(pd.merge, on = 'placeID'), [geo, rest_cuisine, c, rest_parking, rest_payment])

test = rest_payment.join(rating.set_index('placeID'), on = 'placeID')
'test'
df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]






def create_list(new_list, *items):
    for item in items:
        new_list.append(item)
  
new_list = []      
create_list(new_list, 'geo','rest')

new2 = join_tables('placeID', [geo, rest_cuisine, rest_hours, rest_parking, rest_payment])


import pandas as pd
import functools
def join_tables(key_value, *items):
    functools.reduce(functools.partial(pd.merge, on = key_value), [items])

