import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('joined_data.csv')
#dataset = dataset.apply(lambda x: x.astype(str).str.lower())

pivot = pd.DataFrame(dataset.iloc[:,[0,25,26,27]].values)
pivot.columns = ['placeID', 'rating', 'food_rating', 'service_rating']
piv = pd.pivot_table(pivot, values = ['rating', 'food_rating', 'service_rating'], index = 'placeID', aggfunc = 'mean')
pivot.describe()

#test
n_132732 = pivot[pivot['placeID']==132732]
x = n_132732['rating'].sum()

correlation = piv.corr()

dataset['city'].unique()