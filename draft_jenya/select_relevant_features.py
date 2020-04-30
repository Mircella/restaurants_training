import pandas as pd
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.data_frames_cleaning_functions import concatenate_tables
from utils.data_frames_cleaning_functions import extract_restaurants_with_ratings
from utils.data_frames_cleaning_functions import add_missing_restaurants_with_ratings
from draft_jenya.relevant_features_analysis import encode_data_frame

restaurant_payment_types = pd.read_csv('../draft_sveta/no_similiar_categories/restaurant_payment.csv')
restaurant_cuisine_types = pd.read_csv('../draft_sveta/no_similiar_categories/restaurant_cuisine.csv')
restaurant_working_hours = pd.read_csv('../draft_sveta/no_similiar_categories/restaurant_hours.csv')
restaurant_parking_types = pd.read_csv('../draft_sveta/no_similiar_categories/restaurant_parking.csv')
restaurant_geo_places = pd.read_csv('../draft_sveta/no_similiar_categories/restaurant_general.csv')
restaurant_ratings = pd.read_csv('../clean_data/restaurant_ratings.csv')
restaurant_geo_places = restaurant_geo_places.filter(
    ['placeID',
     'alcohol',
     # 'smoking_area',
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

restaurant_payment_types_encoded = encode_data_frame(restaurant_payment_types)
restaurant_payment_types_encoded = restaurant_payment_types_encoded.groupby(restaurant_payment_types_encoded.index).sum()
restaurant_payment_types_and_ratings_encoded_without_index = extract_restaurants_with_ratings(restaurant_payment_types_encoded, restaurant_ratings).reset_index()
restaurant_payment_types_and_ratings_encoded_without_index.rename(columns={'index':'placeID'}, inplace=True)
write_df_to_csv('clean_data_encoded', 'restaurant_payment_types_and_ratings_encoded.csv', restaurant_payment_types_and_ratings_encoded_without_index)

general_rating_restaurant_place_ids = restaurant_ratings.groupby('placeID')['service_rating'].mean().reset_index()
# general_rating_restaurant_place_ids = restaurant_ratings[['placeID','rating']]
buffer = pd.merge(left=general_rating_restaurant_place_ids, right=restaurant_payment_types_and_ratings_encoded_without_index, on='placeID', how="left")

restaurant_cuisine_types_encoded = encode_data_frame(restaurant_cuisine_types)
restaurant_cuisine_types_encoded = restaurant_cuisine_types_encoded.groupby(restaurant_cuisine_types_encoded.index).sum()
restaurant_cuisine_types_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_cuisine_types_encoded, restaurant_ratings).reset_index()
restaurant_cuisine_types_and_ratings_encoded.rename(columns={'index':'placeID'}, inplace=True)
write_df_to_csv('clean_data_encoded', 'restaurant_cuisine_types_and_ratings_encoded.csv', restaurant_cuisine_types_and_ratings_encoded)

buffer = pd.merge(left=buffer, right=restaurant_cuisine_types_and_ratings_encoded, on='placeID', how="left")

restaurant_parking_types_encoded = encode_data_frame(restaurant_parking_types)
restaurant_parking_types_encoded = restaurant_parking_types_encoded.groupby(restaurant_parking_types_encoded.index).sum()
restaurant_parking_types_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_parking_types_encoded, restaurant_ratings).reset_index()
restaurant_parking_types_and_ratings_encoded.rename(columns={'index':'placeID'}, inplace=True)
write_df_to_csv('clean_data_encoded', 'restaurant_parking_types_and_ratings_encoded.csv', restaurant_parking_types_and_ratings_encoded)

buffer = pd.merge(left=buffer, right=restaurant_parking_types_and_ratings_encoded, on='placeID', how="left")

restaurant_geo_places_encoded = encode_data_frame(restaurant_geo_places)
restaurant_geo_places_encoded = restaurant_geo_places_encoded.groupby(restaurant_geo_places_encoded.index).sum()
restaurant_geo_places_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_geo_places_encoded, restaurant_ratings).reset_index()
restaurant_geo_places_and_ratings_encoded.rename(columns={'index':'placeID'}, inplace=True)
write_df_to_csv('clean_data_encoded', 'restaurant_geo_places_and_ratings_encoded.csv', restaurant_geo_places_and_ratings_encoded)

buffer = pd.merge(left=buffer, right=restaurant_geo_places_and_ratings_encoded, on='placeID', how="left")

buffer = buffer.fillna(0)
write_df_to_csv('clean_data_encoded', 'all_restaurant_data_encoded.csv', buffer)
print(buffer.shape)
