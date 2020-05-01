import pandas as pd
import numpy as np

from utils.utils_for_files_storing_and_reading import write_df_to_csv

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_data_frame(data_frame):

    placeIds = data_frame['placeID'].values
    # Selecting categorical features (I intend they are not of numerical type, thus I am choosing only of type object)
    data_frame = data_frame.select_dtypes(include=['object'])

    # Preparing column names for future encoded data frame which will be names of all values of categorical features
    column_names = []
    for column_name in data_frame.columns:
        unique_row_names_sorted = sorted(data_frame[column_name].unique())
        unique_row_names_sorted = [f"{column_name}_{row_name}" for row_name in unique_row_names_sorted]
        column_names.append(unique_row_names_sorted)
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
    result_encoded_data_frame = pd.DataFrame(result_encoded_array, columns=column_names, index=placeIds)
    return result_encoded_data_frame

dataset = pd.read_csv('../../data/concatenated_restaurant_data.csv')
clean_data_set = dataset.drop(['the_geom_meter', 'name', 'address', 'city',
                      'state','country','fax','url','zip',
                      'Rambience','franchise','other_services',
                      'userID','dress_code'], axis = 1)


result_df = encode_data_frame(clean_data_set)
write_df_to_csv('', 'encoded_restaurants_with_ratings_data_all.csv', result_df)
print(result_df.head())