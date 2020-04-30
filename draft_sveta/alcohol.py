import pandas as pd
from utils.data_frames_cleaning_functions import concatenate_tables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


r_general = pd.read_csv('clean_data/restaurant_geo_places.csv')
u_general = pd.read_csv('data/userprofile.csv', delimiter = ';')
rating = pd.read_csv('clean_data/restaurant_ratings.csv')

r_alcohol = r_general[['placeID','alcohol']]
u_alcohol = u_general[['userID','drink_level']]

r_alcohol['alcohol'].unique()
u_alcohol['drink_level'].unique()

general_rating = rating.groupby('placeID')['rating'].mean().reset_index()
alcohol_united = pd.concat([r_alcohol, u_alcohol], axis=1)
test = pd.merge(alcohol_united, general_rating, how = 'left', on = 'placeID')

clean_alcohol = test.drop(['userID'], axis = 1)

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

encode_data_frame(clean_alcohol)
