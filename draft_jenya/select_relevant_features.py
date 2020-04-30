import pandas as pd
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.data_frames_cleaning_functions import concatenate_tables
from utils.data_frames_cleaning_functions import extract_restaurants_with_ratings
from draft_jenya.relevant_features_analysis import encode_data_frame

restaurant_payment_types = pd.read_csv('../clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('../clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('../clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('../clean_data/restaurant_parking_types.csv')
restaurant_geo_places = pd.read_csv('../clean_data/restaurant_geo_places.csv')
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
restaurant_payment_types_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_payment_types_encoded, restaurant_ratings)
write_df_to_csv('clean_data_encoded', 'restaurant_payment_types_and_ratings_encoded.csv', restaurant_payment_types_and_ratings_encoded.reset_index())

restaurant_cuisine_types_encoded = encode_data_frame(restaurant_cuisine_types)
restaurant_cuisine_types_encoded = restaurant_cuisine_types_encoded.groupby(restaurant_cuisine_types_encoded.index).sum()
restaurant_cuisine_types_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_cuisine_types_encoded, restaurant_ratings)
write_df_to_csv('clean_data_encoded', 'restaurant_cuisine_types_and_ratings_encoded.csv', restaurant_cuisine_types_and_ratings_encoded.reset_index())

restaurant_parking_types_encoded = encode_data_frame(restaurant_parking_types)
restaurant_parking_types_encoded = restaurant_parking_types_encoded.groupby(restaurant_parking_types_encoded.index).sum()
restaurant_parking_types_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_parking_types_encoded, restaurant_ratings)
write_df_to_csv('clean_data_encoded', 'restaurant_parking_types_and_ratings_encoded.csv', restaurant_parking_types_and_ratings_encoded.reset_index())

restaurant_geo_places_encoded = encode_data_frame(restaurant_geo_places)
restaurant_geo_places_encoded = restaurant_geo_places_encoded.groupby(restaurant_geo_places_encoded.index).sum()
restaurant_geo_places_and_ratings_encoded = extract_restaurants_with_ratings(restaurant_geo_places_encoded, restaurant_ratings)
write_df_to_csv('clean_data_encoded', 'restaurant_geo_places_and_ratings_encoded.csv', restaurant_geo_places_and_ratings_encoded)

payments_and_cuisines = concatenate_tables(
    restaurant_payment_types_and_ratings_encoded,
    restaurant_cuisine_types_and_ratings_encoded,
    restaurant_parking_types_and_ratings_encoded,
    restaurant_geo_places_and_ratings_encoded
)
write_df_to_csv(payments_and_cuisines)
print(payments_and_cuisines.shape)
