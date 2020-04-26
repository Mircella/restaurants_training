import pandas as pd
from utils.data_frames_cleaning_functions import merge_and_group
from draft_jenya.restaurants_data_preprocessing import all_restaurant_ids


restaurant_payment_types = pd.read_csv('../clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('../clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('../clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('../clean_data/restaurant_parking_types.csv')
restaurant_ratings = pd.read_csv('../clean_data/restaurant_ratings.csv')
restaurant_geo_places = pd.read_csv('../clean_data/restaurant_geo_places.csv')

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

# merge_and_group(
#     left_df=restaurant_ratings,
#     right_df=restaurant_parking_types,
#     merge_column='placeID',
#     group_column='parking_lot',
#     estimate_column='service_rating'
# )

merge_and_group(
    left_df=restaurant_ratings,
    right_df=restaurant_parking_types,
    merge_column='placeID',
    group_column='parking_lot',
    estimate_column='rating'
)

# merge_and_group(
#     left_df=restaurant_ratings,
#     right_df=restaurant_payment_types,
#     merge_column='placeID',
#     group_column='Rpayment',
#     estimate_column='service_rating'
# )

# # Group by the parking_lot column
# cuisine_group = df.groupby("Rcuisine")
#
# # Calculate the mean ratings
# print(cuisine_group["food_rating"].mean())
# print(cuisine_group['food_rating'].describe())