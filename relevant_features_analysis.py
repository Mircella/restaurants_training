import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from file_utils import write_df_to_csv
from restaurants_data_preprocessing import all_restaurant_ids

restaurant_payment_types = pd.read_csv('clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('clean_data/restaurant_parking_types.csv')
restaurant_geo_places = pd.read_csv('clean_data/restaurant_geo_places.csv')
restaurant_ratings = pd.read_csv('clean_data/restaurant_ratings.csv')
# Finding features that have impact on restaurant's rating
encoded_restaurant_cuisines = pd.get_dummies(restaurant_cuisine_types, columns=['Rcuisine'])

# remove duplicate restaurant ID's.
# A restaurant with multiple cuisine categories would have multiple columns equal 1
encoded_restaurant_cuisines = encoded_restaurant_cuisines.groupby('placeID', as_index=False).sum()
write_df_to_csv(data_dir="relevant_features", file_name="encoded_restaurant_cuisines.csv", data_frame=encoded_restaurant_cuisines)

# use dummy variables for different cuisine categories of the restaurants
encoded_restaurant_parking = pd.get_dummies(restaurant_parking_types, columns=['parking_lot'])
write_df_to_csv(data_dir="relevant_features", file_name="encoded_restaurant_parking.csv", data_frame=encoded_restaurant_parking)

# remove duplicate restaurant ID's.
# A restaurant with multiple parking options would have multiple columns equal 1
encoded_restaurant_parking = encoded_restaurant_parking.groupby('placeID', as_index=False).sum()
write_df_to_csv(data_dir="relevant_features", file_name="res_parking_1.csv", data_frame=encoded_restaurant_parking)

# filter only thos columns from geo places which has potentially impact on the rating of users
restaurant_geo_places = restaurant_geo_places.filter(
    ['placeID',
     'alcohol',
     'smoking_area',
     'dress_code',
     'accessibility',
     'price',
     'Rambience',
     'franchise',
     'area',
     'other_services'],
    axis=1
)


df_res = pd.DataFrame({'placeID': all_restaurant_ids})
df_res = pd.merge(left=df_res, right=encoded_restaurant_cuisines, how="left", on="placeID")
df_res = pd.merge(left=df_res, right=encoded_restaurant_parking, how="left", on="placeID")
df_res = pd.merge(left=df_res, right=restaurant_geo_places, how="left", on="placeID")

write_df_to_csv(data_dir="relevant_features", file_name="df_res.csv", data_frame=df_res)

