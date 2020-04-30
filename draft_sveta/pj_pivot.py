import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string


dataset = pd.read_csv('../clean_data/restaurant_geo_places.csv')

dataset = dataset.apply(lambda x: x.astype(str).str.lower())
cities = dataset['city'].unique()

pivot = pd.DataFrame(dataset.iloc[:,[0,25,26,27]].values)
pivot.columns = ['placeID', 'rating', 'food_rating', 'service_rating']
piv = pd.pivot_table(pivot, values = ['rating', 'food_rating', 'service_rating'], index = 'placeID', aggfunc = 'mean')
pivot.describe()
cities_1 = ['s.l.p.', 'san luis potosi', 'slp', 'san luis potos','san luis potosi ','s.l.p']
print(dataset['city'].unique())
condition = dataset.city.isin(cities_1)
filtered = dataset[condition]
print(dataset.head())
dataset.city=dataset.city.apply(lambda x:''.join([i for i in x
                            if i not in string.punctuation]))
dataset.city=dataset.city.replace(['','san luis potos','san luis potosi','slp','san luis potosi '],'san luis potosi' )
dataset.city=dataset.city.replace(['victoria','cd victoria','victoria '],'ciudad victoria' )
print(dataset.city.value_counts())
# pivot = pd.DataFrame(dataset.iloc[:,[0,25,26,27]].values)
# pivot.columns = ['placeID', 'rating', 'food_rating', 'service_rating']
# piv = pd.pivot_table(pivot, values = ['rating', 'food_rating', 'service_rating'], index = 'placeID', aggfunc = 'mean')
# pivot.describe()
#
#test
n_132732 = pivot[pivot['placeID']==132732]
x = n_132732['rating'].sum()

correlation = piv.corr()
dataset['city'].unique()
print(correlation)
