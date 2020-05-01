import pandas as pd
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.data_frames_cleaning_functions import encode_data_frame_and_group_by_index
from utils.data_frames_cleaning_functions import inner_join_data_frames_by_column
from utils.data_frames_cleaning_functions import move_index_to_column


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
# general_rating_restaurant_place_ids = restaurant_ratings.groupby('placeID')['rating'].mean().reset_index()
general_rating_restaurant_place_ids = restaurant_ratings[['placeID','rating']]
# I have done the same encoding and merging with restaurants ratings for restaurant_geo_places
restaurant_geoplaces = encode_data_frame_and_group_by_index(restaurant_geo_places)
restaurant_geoplaces = inner_join_data_frames_by_column(restaurant_geoplaces, restaurant_ratings, 'placeID').reset_index()
restaurant_geoplaces = move_index_to_column(restaurant_geoplaces, 'placeID')
write_df_to_csv('.', 'restaurant_geo_places_and_ratings_encoded.csv', restaurant_geoplaces)
restaurants = pd.merge(left=general_rating_restaurant_place_ids, right=restaurant_geoplaces, on='placeID', how="left")
restaurant_geoplaces_ids = restaurant_geoplaces['placeID'].values
g = general_rating_restaurant_place_ids[general_rating_restaurant_place_ids['placeID'].isin(restaurant_geoplaces_ids)]
# Here I replaced NaN values in result data frame with zero so it means that restaurant is missing information in this type of category
restaurants = restaurants.fillna(0)
write_df_to_csv('../../clean_data_encoded', 'restaurant_geo_places.csv', restaurants)
print(restaurants.shape)
