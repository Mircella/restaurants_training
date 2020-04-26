import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from prep_func import join_tables
from prep_func import concatenate_tables
from prep_func import drop_duplicated_rows_and_columns
from prep_func import drop_nan
from prep_func import find_unique_records_number_by_column
from file_utils import write_df_to_csv

# loading restaurants data
restaurant_payment_types = pd.read_csv('data/chefmozaccepts.csv', delimiter =';')
write_df_to_csv('clean_data', 'restaurant_payment_types.csv', restaurant_payment_types)

restaurant_cuisine_types = pd.read_csv('data/chefmozcuisine.csv', delimiter =';')
write_df_to_csv('clean_data', 'restaurant_cuisine_types.csv', restaurant_cuisine_types)

restaurant_working_hours = pd.read_csv('data/chefmozhours.csv', delimiter =',')

for i, series in restaurant_working_hours.iterrows():
    hours = restaurant_working_hours.loc[i, "hours"][0:len(restaurant_working_hours.loc[i, "hours"]) - 1]
    restaurant_working_hours.loc[i, "hours"] = hours
    hours = restaurant_working_hours.loc[i, "days"][0:len(restaurant_working_hours.loc[i, "days"]) - 1]
    restaurant_working_hours.loc[i, "days"] = hours

write_df_to_csv('clean_data', 'restaurant_working_hours.csv', restaurant_working_hours)

restaurant_parking = pd.read_csv('data/chefmozparking.csv', delimiter =';')
write_df_to_csv('clean_data', 'restaurant_parking_types.csv', restaurant_parking)

restaurant_geo_places = pd.read_csv('data/geoplaces.csv', delimiter =';', encoding='latin-1')
write_df_to_csv('clean_data', 'restaurant_geo_places.csv', restaurant_geo_places)

restaurant_ratings = pd.read_csv('data/rating_final.csv', delimiter =';')
write_df_to_csv('clean_data', 'restaurant_ratings.csv', restaurant_ratings)

# Extracting how many payments types exist in restaurants
print(f"Number of unique restaurants with payment type specified:{len(restaurant_payment_types['placeID'].unique())}")
print(f"Number of payment types:{len(restaurant_payment_types['Rpayment'].unique())}")

# Extracting how many cuisine types exist in restaurants
print(f"Number of unique restaurants with cuisine type specified:{len(restaurant_cuisine_types['placeID'].unique())}")
print(f"Number of cuisine types:{len(restaurant_cuisine_types['Rcuisine'].unique())}")

# Extracting how many parking types exist in restaurants
print(f"Number of unique restaurants with parking specified:{len(restaurant_parking['placeID'].unique())}")
print(f"Parking types:{restaurant_parking['parking_lot'].unique()}")
print(f"Number of parking types:{len(restaurant_parking['parking_lot'].unique())}")

# Extracting how many restaurants was evaluated by users
estimated_restaurant_ids = restaurant_ratings['placeID'].unique()
print(f"Number of restaurants evaluated by users:{len(estimated_restaurant_ids)}")

# Extracting how many restaurants have descriptions in geoplaces file
print(f"Number of restaurants that have description:{len(restaurant_geo_places['placeID'].unique())}")

# Extracting how many restaurants published their working hours
print(f"Number of restaurants with specified working hours:{len(restaurant_working_hours['placeID'].unique())}")

# How many restaurants do we have across all restaurants data files
all_restaurant_ids = find_unique_records_number_by_column(
    'placeID',
    restaurant_geo_places,
    restaurant_cuisine_types,
    restaurant_working_hours,
    restaurant_parking,
    restaurant_payment_types
)

print(f"All ids of restaurants: {len(all_restaurant_ids)}")

# joining data of restaurants from all tables by their place id to exclude restaurants that do not have any data and will not have impact on the model
joined_restaurant_data = join_tables(
    'placeID',
    restaurant_geo_places,
    restaurant_cuisine_types,
    restaurant_working_hours,
    restaurant_parking,
    restaurant_payment_types,
    restaurant_ratings
)

# see how many records we have after joining
print(f"Number of joined records:{len(joined_restaurant_data)}")

# drop nan
joined_restaurant_data = drop_nan(joined_restaurant_data)
# see how many records we have after dropping NaN
print(f"Number of joined records after dropping NaN values:{len(joined_restaurant_data)}")

# drop duplicated rows and columns
joined_restaurant_data = drop_duplicated_rows_and_columns(joined_restaurant_data)
# see how many records we have after dropping duplicated columns and rows
print(f"Number of joined records after dropping duplicated columns and rows:{len(joined_restaurant_data)}")

# write joined restaurant data frame to csv file
write_df_to_csv(data_dir="data", file_name="joined_restaurant_data.csv", data_frame=joined_restaurant_data)

# concatenating data of restaurants from all tables by their place id to exclude restaurants that do not have any data and will not have impact on the model
concatenated_restaurant_data = concatenate_tables(
    restaurant_geo_places,
    restaurant_cuisine_types,
    restaurant_working_hours,
    restaurant_parking,
    restaurant_payment_types,
    restaurant_ratings
)

# see how many records we have after concatenation
print(f"Number of concatenated records:{len(concatenated_restaurant_data)}")

# # drop nan
# concatenated_restaurant_data = drop_nan(concatenated_restaurant_data)
# see how many records we have after dropping NaN
print(f"Number of concatenated records after dropping NaN values:{len(concatenated_restaurant_data)}")

# drop duplicated rows and columns
concatenated_restaurant_data = drop_duplicated_rows_and_columns(concatenated_restaurant_data)

# see how many records we have after dropping duplicated columns and rows
print(f"Number of concatenated records after dropping duplicated columns and rows:{len(concatenated_restaurant_data)}")

# write concatenated restaurant data frame to csv file
write_df_to_csv(data_dir="data", file_name="concatenated_restaurant_data.csv", data_frame=concatenated_restaurant_data)
