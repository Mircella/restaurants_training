import pandas as pd
import numpy as np
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.data_frames_cleaning_functions import join_tables
from utils.data_frames_cleaning_functions import find_unique_records_number_by_column

restaurant_payment_types = pd.read_csv('../clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('../clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('../clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('../clean_data/restaurant_parking_types.csv')
restaurant_geo_places = pd.read_csv('../clean_data/restaurant_geo_places.csv')
restaurant_ratings = pd.read_csv('../clean_data/restaurant_ratings.csv')

# Finding features that have impact on restaurant's rating
# encoded_restaurant_cuisines = pd.get_dummies(restaurant_cuisine_types, columns=['Rcuisine'])

# A restaurant with multiple cuisine categories would have multiple columns equal 1
# encoded_restaurant_cuisines = encoded_restaurant_cuisines.groupby('placeID', as_index=False).sum()
# write_df_to_csv(data_dir="relevant_features", file_name="encoded_restaurant_cuisines.csv", data_frame=encoded_restaurant_cuisines)

# use dummy variables for different cuisine categories of the restaurants
# encoded_restaurant_parking = pd.get_dummies(restaurant_parking_types, columns=['parking_lot'])
# write_df_to_csv(data_dir="relevant_features", file_name="encoded_restaurant_parking.csv", data_frame=encoded_restaurant_parking)

# remove duplicate restaurant ID's.
# A restaurant with multiple parking options would have multiple columns equal 1
# encoded_restaurant_parking = encoded_restaurant_parking.groupby('placeID', as_index=False).sum()
# write_df_to_csv(data_dir="relevant_features", file_name="res_parking_1.csv", data_frame=encoded_restaurant_parking)

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

joined_by_place_id = join_tables('placeID', restaurant_cuisine_types, restaurant_payment_types, restaurant_parking_types, restaurant_geo_places)

# How many restaurants do we have across all restaurants data files
all_restaurant_ids = find_unique_records_number_by_column(
    'placeID',
    restaurant_geo_places,
    restaurant_cuisine_types,
    restaurant_working_hours,
    restaurant_parking_types,
    restaurant_payment_types
)

place_ids_and_parking_types_merged = pd.DataFrame({'placeID': all_restaurant_ids})
place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_parking_types, how="left", on="placeID")
place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_payment_types, how="left", on="placeID")
place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_working_hours, how="left", on="placeID")
place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_cuisine_types, how="left", on="placeID")
place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_geo_places, how="left", on="placeID")
place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_ratings, how='left', on="placeID")

# write_df_to_csv(data_dir="../data", file_name="place_ids_and_parking_types_merged.csv", data_frame=place_ids_and_parking_types_merged)
write_df_to_csv(data_dir="../data", file_name="joined_by_place_id.csv", data_frame=joined_by_place_id)
