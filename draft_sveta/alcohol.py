import pandas as pd
import numpy as np
from utils.data_frames_cleaning_functions import concatenate_tables
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.data_frames_cleaning_functions import extract_restaurants_with_ratings

r_general = pd.read_csv('clean_data/restaurant_geo_places.csv')
u_general = pd.read_csv('data/userprofile.csv', delimiter = ';')
rating = pd.read_csv('clean_data/restaurant_ratings.csv')

r_alcohol = r_general[['placeID','alcohol']]
u_alcohol = u_general[['userID','drink_level']]

r_alcohol['alcohol'].unique()
u_alcohol['drink_level'].unique()

general_rating = rating[['placeID', 'userID','rating']]
rating_merged_rest = pd.merge(general_rating, r_alcohol, how = 'left', on = 'placeID')
rating_merged_user = pd.merge(rating_merged_rest , u_alcohol, how = 'left', on = 'userID')


write_df_to_csv('user_rest_data','alcohol.csv',rating_merged_user)

#encoding
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

dataset = pd.read_csv('user_rest_data/alcohol.csv')
clean_data_set = dataset.drop(['userID'], axis = 1)
t = clean_data_set.groupby(['placeID', 'alcohol', 'drink_level'])['rating'].mean().reset_index()


X = t.iloc[:,1:3].values
y = t.iloc[:,3].values

labelencoder_X = LabelEncoder()
t['alcohol'] = labelencoder_X.fit_transform(t['alcohol'])
X[:,0] = labelencoder_X.fit_transform(X[:,0])
labelencoder_X_1 = LabelEncoder()
t['drink_level'] = labelencoder_X_1.fit_transform(t['drink_level'])
X[:,1] = labelencoder_X.fit_transform(X[:,1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


#decision_tree
from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(random_state = 0)
regressor_tree.fit(X_train, y_train)
y_pred = regressor_tree.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


#randon_forect
from sklearn.ensemble import RandomForestRegressor
regressor_forest = RandomForestRegressor(n_estimators = 300, random_state =0)
regressor_forest.fit(X_train, y_train)
r2_score(y_test, y_pred)
