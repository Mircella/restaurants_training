import pandas as pd
import numpy as np
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.data_frames_cleaning_functions import join_tables
from utils.data_frames_cleaning_functions import find_unique_records_number_by_column
from draft_jenya.relevant_features_analysis import encode_data_frame

restaurant_payment_types = pd.read_csv('../clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('../clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('../clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('../clean_data/restaurant_parking_types.csv')
restaurant_geo_places = pd.read_csv('../clean_data/restaurant_geo_places.csv')
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
joined_by_place_id_encoded = encode_data_frame(restaurant_payment_types)
joined_by_place_id_encoded = joined_by_place_id_encoded.groupby(joined_by_place_id_encoded.index).sum()
# How many restaurants do we have across all restaurants data files
# all_restaurant_ids = find_unique_records_number_by_column(
#     'placeID',
#     restaurant_geo_places,
#     restaurant_cuisine_types,
#     restaurant_working_hours,
#     restaurant_parking_types,
#     restaurant_payment_types
# )
#
# place_ids_and_parking_types_merged = pd.DataFrame({'placeID': all_restaurant_ids})
# place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_parking_types, how="left", on="placeID")
# place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_payment_types, how="left", on="placeID")
# place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_working_hours, how="left", on="placeID")
# place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_cuisine_types, how="left", on="placeID")
# place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_geo_places, how="left", on="placeID")
# place_ids_and_parking_types_merged = pd.merge(left=place_ids_and_parking_types_merged, right=restaurant_ratings, how='left', on="placeID")

# write_df_to_csv(data_dir="../data", file_name="place_ids_and_parking_types_merged.csv", data_frame=place_ids_and_parking_types_merged)
# write_df_to_csv(data_dir="../data", file_name="joined_by_place_id.csv", data_frame=joined_by_place_id)
