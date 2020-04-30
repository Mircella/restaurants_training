import pandas as pd
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.data_frames_cleaning_functions import encode_data_frame_and_group_by_index
from utils.data_frames_cleaning_functions import inner_join_data_frames_by_column
from utils.data_frames_cleaning_functions import move_index_to_column


restaurant_payment_types = pd.read_csv('../../clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('../../clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('../../clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('../../clean_data/restaurant_parking_types.csv')
restaurant_geo_places = pd.read_csv('../../clean_data/restaurant_geo_places.csv')
restaurant_ratings = pd.read_csv('../../clean_data/restaurant_ratings.csv')
restaurant_geo_places = restaurant_geo_places.filter(
    ['placeID',
     'alcohol',
     'dress_code',
     'accessibility',
     'price',
     'Rambience',
     'franchise',
     'area',
     'other_services'
     ],
    axis=1
)
general_rating_restaurant_place_ids = restaurant_ratings.groupby('placeID')['rating'].mean().reset_index()

# I encode all categorical features by putting categories as column names and
# adding to each column name prefix which is type of category to easy recognize which category this column name belongs
# to, for ex: Rpayment_Visa
restaurant_payments = encode_data_frame_and_group_by_index(restaurant_payment_types)

# I found intersection of encoded payment data with restaurants which have ratings by place id
restaurant_payments = inner_join_data_frames_by_column(restaurant_payments, restaurant_ratings, 'placeID')

# I moved index to column to display it in csv file (for more convenient navigation)
# and also to be able to do merge between tables by this columns
restaurant_payments = move_index_to_column(restaurant_payments, 'placeID', True)

# I saved this encoded data to csv file for future analysis
write_df_to_csv('', 'restaurant_payment_types_and_ratings_encoded.csv', restaurant_payments)

# I made left join of restaurant ratings with restaurant_payments_encoded so that I always have 130 restaurants with ratings
# and some of them can probably miss pinformation about payments
# but it is also probable that after following merging with other data frames they will have information in other categories
# that can potentially impact the rating
restaurants = pd.merge(left=general_rating_restaurant_place_ids, right=restaurant_payments, on='placeID', how="left")

# I have done the same encoding and merging with restaurants ratings for restaurant_parking_types
restaurant_parkings = encode_data_frame_and_group_by_index(restaurant_parking_types)
restaurant_parkings = inner_join_data_frames_by_column(restaurant_parkings, restaurant_ratings, 'placeID').reset_index()
restaurant_parkings = move_index_to_column(restaurant_parkings, 'placeID')
write_df_to_csv('', 'restaurant_parking_types_and_ratings_encoded.csv', restaurant_parkings)

restaurants = pd.merge(left=restaurants, right=restaurant_parkings, on='placeID', how="left")

# I have done the same encoding and merging with restaurants ratings for restaurant_cuisine_types
restaurant_cuisines = encode_data_frame_and_group_by_index(restaurant_cuisine_types)
restaurant_cuisines = inner_join_data_frames_by_column(restaurant_cuisines, restaurant_ratings, 'placeID').reset_index()
restaurant_cuisines = move_index_to_column(restaurant_cuisines, 'placeID')
write_df_to_csv('', 'restaurant_cuisine_types_and_ratings_encoded.csv', restaurant_cuisines)

restaurants = pd.merge(left=restaurants, right=restaurant_cuisines, on='placeID', how="left")

# I have done the same encoding and merging with restaurants ratings for restaurant_geo_places
restaurant_geoplaces = encode_data_frame_and_group_by_index(restaurant_geo_places)
restaurant_geoplaces = inner_join_data_frames_by_column(restaurant_geoplaces, restaurant_ratings, 'placeID').reset_index()
restaurant_geoplaces = move_index_to_column(restaurant_geoplaces, 'placeID')
write_df_to_csv('', 'restaurant_geo_places_and_ratings_encoded.csv', restaurant_geoplaces)

restaurants = pd.merge(left=restaurants, right=restaurant_geoplaces, on='placeID', how="left")

# Here I replaced NaN values in result data frame with zero so it means that restaurant is missing information in this type of category
restaurants = restaurants.fillna(0)
write_df_to_csv('', 'all_restaurant_data_encoded.csv', restaurants)
print(restaurants.shape)
