import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from prep_func import join_tables
from prep_func import concatenate_tables
from prep_func import drop_duplicated_rows_and_columns
from prep_func import drop_nan
from prep_func import find_unique_records_number_by_column
from file_utils import write_df_to_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from restaurants_data_preprocessing import all_restaurant_ids


restaurant_payment_types = pd.read_csv('clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('clean_data/restaurant_parking_types.csv')
restaurant_ratings = pd.read_csv('clean_data/restaurant_ratings.csv')
restaurant_geo_places = pd.read_csv('clean_data/restaurant_geo_places.csv')

df = pd.DataFrame({'placeID': all_restaurant_ids})
# These are the ones that I think might be relevant
restaurant_features = restaurant_geo_places[['placeID', 'alcohol', 'smoking_area', 'other_services', 'price', 'dress_code',
               'accessibility','area']]
# Merge the dataframes

df = pd.merge(left=df, right=restaurant_parking_types, on="placeID", how="left")
df = pd.merge(left=df, right=restaurant_cuisine_types, how="left", on="placeID")
df = pd.merge(left=df, right=restaurant_payment_types, how="left", on="placeID")
df = pd.merge(left=df, right=restaurant_ratings, how="left", on="placeID")
df = pd.merge(left=df, right=restaurant_features, how="left", on="placeID")

df_parking_and_rating = pd.merge(left=restaurant_ratings, right=restaurant_parking_types, on="placeID", how="left")
# Show the merged data
print(df_parking_and_rating.head())
# Group by the parking_lot column
parking_group = df_parking_and_rating.groupby("parking_lot")

# Calculate the mean ratings
print(parking_group['service_rating'].describe())

# # Group by the parking_lot column
# cuisine_group = df.groupby("Rcuisine")
#
# # Calculate the mean ratings
# print(cuisine_group["food_rating"].mean())
# print(cuisine_group['food_rating'].describe())