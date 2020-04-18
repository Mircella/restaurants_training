import pandas as pd
import functools


def join_tables(key_value, *data_frames):
    joined = functools.reduce(functools.partial(pd.merge, on = key_value), data_frames[0])
    return joined

rating = pd.read_csv('data/rating_final.csv', delimiter = ';')
rest_payment = pd.read_csv('data/chefmozaccepts.csv', delimiter = ';')
rest_cuisine = pd.read_csv('data/chefmozcuisine.csv', delimiter = ';')
rest_hours = pd.read_csv('data/chefmozhours.csv', delimiter = ',')
rest_parking = pd.read_csv('data/chefmozparking.csv', delimiter = ';')
geo = pd.read_csv('data/geoplaces.csv', delimiter = ';',encoding='latin-1')

new_df = join_tables('placeID', [geo, rest_cuisine, rest_hours, rest_parking, rest_payment, rating])
print(len(new_df))
print(new_df)