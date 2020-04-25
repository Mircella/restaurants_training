import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from prep_func import join_tables
from prep_func import drop_nan
from file_utils import write_df_to_csv

#stats tables
rating = pd.read_csv('data/rating_final.csv', delimiter = ';')
rest_payment = pd.read_csv('data/chefmozaccepts.csv', delimiter = ';')
rest_cuisine = pd.read_csv('data/chefmozcuisine.csv', delimiter = ';')
rest_hours = pd.read_csv('data/chefmozhours.csv', delimiter = ',')
rest_parking = pd.read_csv('data/chefmozparking.csv', delimiter = ';')
geo = pd.read_csv('data/geoplaces.csv', delimiter = ';',encoding='latin-1')


joined = join_tables('placeID', [geo, rest_cuisine, rest_hours, rest_parking, rest_payment, rating])
print(len(joined))
print(joined)

#drop nan
joined = drop_nan(joined)

write_df_to_csv('data','joined_data.csv',joined)

