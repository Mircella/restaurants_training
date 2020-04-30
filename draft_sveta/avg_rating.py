#rating average
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_frames_cleaning_functions import join_tables
from data_frames_cleaning_functions import concatenate_tables

#rest data
r_payment = pd.read_csv('data/chefmozaccepts.csv', delimiter = ';')
r_cuisine = pd.read_csv('data/chefmozcuisine.csv', delimiter = ';')
r_hours = pd.read_csv('data/chefmozhours.csv', delimiter = ',')
r_parking = pd.read_csv('data/chefmozparking.csv', delimiter = ';')
r_general = pd.read_csv('data/chefmozaccepts.csv', delimiter = ';', encoding='latin-1')
rating = pd.read_csv('data/rating_final.csv', delimiter =';')

unique_ratings = rating['placeID'].unique()

r_joined = join_tables('placeID',r_payment, r_cuisine, r_hours, r_parking,
                       r_general)

test = concatenate_tables(r_joined, unique_ratings) #here is an error


#test = df_r.groupby('placeID')['rating'].mean() - for mean