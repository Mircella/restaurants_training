import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from file_utils import write_df_to_csv
from restaurants_data_preprocessing import restaurant_cuisine_types
from restaurants_data_preprocessing import restaurant_parking
from restaurants_data_preprocessing import restaurant_geo_places
from restaurants_data_preprocessing import all_restaurant_ids

# Finding features that have impact on restaurant's rating
res_cuisine = pd.get_dummies(restaurant_cuisine_types, columns=['Rcuisine'])
# write_df_to_csv(data_dir="ratings_matrices", file_name="res_cuisine.csv", data_frame=res_cuisine)

# remove duplicate restaurant ID's.
# A restaurant with multiple cuisine categories would have multiple columns equal 1
res_cuisine_1 = res_cuisine.groupby('placeID',as_index=False).sum()
# write_df_to_csv(data_dir="ratings_matrices", file_name="res_cuisine_1.csv", data_frame=res_cuisine_1)

# use dummy variables for different cuisine categories of the restaurants
res_parking = pd.get_dummies(restaurant_parking,columns=['parking_lot'])
# write_df_to_csv(data_dir="ratings_matrices", file_name="res_parking.csv", data_frame=res_parking)

# remove duplicate restaurant ID's.
# A restaurant with multiple parking options would have multiple columns equal 1
res_parking_1 = res_parking.groupby('placeID',as_index=False).sum()
# write_df_to_csv(data_dir="ratings_matrices", file_name="res_parking_1.csv", data_frame=res_parking_1)

res_features = restaurant_geo_places[['placeID','alcohol','smoking_area','other_services','price','dress_code',
               'accessibility','area']]

df_res = pd.DataFrame({'placeID': all_restaurant_ids})
df_res = pd.merge(left=df_res, right=res_cuisine_1, how="left", on="placeID")
df_res = pd.merge(left=df_res, right=res_parking_1, how="left", on="placeID")
df_res = pd.merge(left=df_res, right=res_features, how="left", on="placeID")

# write_df_to_csv(data_dir="ratings_matrices", file_name="df_res.csv", data_frame=df_res)

# encoding categorical data
restaurant_cuisines_array = restaurant_cuisine_types.iloc[:,:].values
cusine_label_encoder = LabelEncoder()
restaurant_cuisines_array[:, 1] = cusine_label_encoder.fit_transform(restaurant_cuisines_array[:, 1])
restaurant_cuisines_encoded = restaurant_cuisines_array

oneHotEncoder = OneHotEncoder('auto')
columnTransformer = ColumnTransformer([('encoder', oneHotEncoder, [1])], remainder='passthrough')
transformation = columnTransformer.fit_transform(restaurant_cuisine_types)
transformation = transformation.astype(float)
dataset = np.array(transformation)
restaurant_cuisines_encoded_df = pd.DataFrame(dataset)
write_df_to_csv(data_dir="ratings_matrices", file_name="restaurant_cuisines_encoded_df.csv", data_frame=restaurant_cuisines_encoded_df)
