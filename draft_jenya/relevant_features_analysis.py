import pandas as pd
import numpy as np

from utils.utils_for_files_storing_and_reading import write_df_to_csv

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_data_frame(data_frame):
    # Selecting categorical features (I intend they are not of numerical type, thus I am choosing only of type object)
    data_frame = data_frame.select_dtypes(include=['object'])

    # Preparing column names for future encoded data frame which will be names of all values of categorical features
    column_names = []
    for column_name in data_frame.columns:
        column_names.append(sorted(data_frame[column_name].unique()))
    column_names = [item for sublist in column_names for item in sublist]

    # Mapping data frame to array to be able to use LabelEncoder and OneHotEncoder
    data_frame_array = data_frame.iloc[:, :].values

    # Encoding values of categorical features with LabelEncoder
    for i in range(0, data_frame_array.shape[1]):
        label_encoder = LabelEncoder()
        data_frame_array[:, i] = label_encoder.fit_transform(data_frame_array[:, i])

    # Encoding categorical features values to be columns of encoded matrix,
    # here I chose all columns to be encoded so that each unique value in this column will become column of encoded matrix
    one_hot_encoder = OneHotEncoder(categorical_features=np.array(range(data_frame_array.shape[1])))
    result_encoded_array = one_hot_encoder.fit_transform(data_frame_array).toarray()

    # Here I create encoded matrix back to data frame which is indexed by restaurant place id and has prepared above column_names
    result_encoded_data_frame = pd.DataFrame(result_encoded_array, columns=column_names, index=np.array(dataset['placeID']))
    return result_encoded_data_frame


# restaurant_payment_types = pd.read_csv('clean_data/restaurant_payment_types.csv')
# restaurant_cuisine_types = pd.read_csv('clean_data/restaurant_cuisine_types.csv')
# restaurant_working_hours = pd.read_csv('clean_data/restaurant_working_hours.csv')
# restaurant_parking_types = pd.read_csv('clean_data/restaurant_parking_types.csv')
# restaurant_geo_places = pd.read_csv('clean_data/restaurant_geo_places.csv')
# restaurant_ratings = pd.read_csv('clean_data/restaurant_ratings.csv')
# # Finding features that have impact on restaurant's rating
# encoded_restaurant_cuisines = pd.get_dummies(restaurant_cuisine_types, columns=['Rcuisine'])
#
# # A restaurant with multiple cuisine categories would have multiple columns equal 1
# encoded_restaurant_cuisines = encoded_restaurant_cuisines.groupby('placeID', as_index=False).sum()
# write_df_to_csv(data_dir="relevant_features", file_name="encoded_restaurant_cuisines.csv", data_frame=encoded_restaurant_cuisines)
#
# # use dummy variables for different cuisine categories of the restaurants
# encoded_restaurant_parking = pd.get_dummies(restaurant_parking_types, columns=['parking_lot'])
# write_df_to_csv(data_dir="relevant_features", file_name="encoded_restaurant_parking.csv", data_frame=encoded_restaurant_parking)
#
# # remove duplicate restaurant ID's.
# # A restaurant with multiple parking options would have multiple columns equal 1
# encoded_restaurant_parking = encoded_restaurant_parking.groupby('placeID', as_index=False).sum()
# write_df_to_csv(data_dir="relevant_features", file_name="res_parking_1.csv", data_frame=encoded_restaurant_parking)
#
# # filter only thos columns from geo places which has potentially impact on the rating of users
# restaurant_geo_places = restaurant_geo_places.filter(
#     ['placeID',
#      'alcohol',
#      'smoking_area',
#      'dress_code',
#      'accessibility',
#      'price',
#      'Rambience',
#      'franchise',
#      'area',
#      'other_services'],
#     axis=1
# )
#
#
# df_res = pd.DataFrame({'placeID': all_restaurant_ids})
# df_res = pd.merge(left=df_res, right=encoded_restaurant_cuisines, how="left", on="placeID")
# df_res = pd.merge(left=df_res, right=encoded_restaurant_parking, how="left", on="placeID")
# df_res = pd.merge(left=df_res, right=restaurant_geo_places, how="left", on="placeID")
#
# write_df_to_csv(data_dir="relevant_features", file_name="df_res.csv", data_frame=df_res)

dataset = pd.read_csv('data/concatenated_restaurant_data.csv')
clean_data_set = dataset.drop(['the_geom_meter', 'name', 'address', 'city',
                      'state','country','fax','url','zip',
                      'Rambience','franchise','other_services',
                      'userID','dress_code'], axis = 1)


result_df = encode_data_frame(clean_data_set)
write_df_to_csv('data', 'encoded_restaurants_with_ratings_data_all.csv',result_df)
print(result_df.head())