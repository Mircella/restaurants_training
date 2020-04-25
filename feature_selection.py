import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from prep_func import join_tables
from prep_func import concatenate_tables
from prep_func import drop_duplicated_rows_and_columns
from prep_func import drop_nan
from prep_func import find_unique_records_number_by_column
from file_utils import write_df_to_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def encode_input_features(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    x_train_enc = oe.transform(X)
    return x_train_enc


def encode_labels(y):
    le = LabelEncoder()
    le.fit(y)
    y_train_enc = le.transform(y)
    return y_train_enc


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


restaurant_payment_types = pd.read_csv('clean_data/restaurant_payment_types.csv')
restaurant_cuisine_types = pd.read_csv('clean_data/restaurant_cuisine_types.csv')
restaurant_working_hours = pd.read_csv('clean_data/restaurant_working_hours.csv')
restaurant_parking_types = pd.read_csv('clean_data/restaurant_parking_types.csv')
restaurant_geo_places = pd.read_csv('clean_data/restaurant_geo_places.csv')
restaurant_ratings = pd.read_csv('clean_data/restaurant_ratings.csv')

restaurant_geo_places = restaurant_geo_places.filter(
    ['alcohol', 'smoking_area', 'dress_code', 'accessibility', 'price', 'area',
     'other_services'],
    axis=1
)

concatenated_restaurant_features = concatenate_tables(
    # restaurant_cuisine_types,
    restaurant_parking_types,
    restaurant_payment_types,
    restaurant_geo_places,
    restaurant_ratings
)

# see how many records we have after concatenation
print(f"Number of concatenated records without geo places & users:{len(concatenated_restaurant_features)}")

# drop nan
concatenated_restaurant_features = drop_nan(concatenated_restaurant_features)
# see how many records we have after dropping NaN
print(f"Number of concatenated records after dropping NaN values:{len(concatenated_restaurant_features)}")

# drop duplicated rows and columns
concatenated_restaurant_features = drop_duplicated_rows_and_columns(concatenated_restaurant_features)

# see how many records we have after dropping duplicated columns and rows
print(f"Number of concatenated records after dropping duplicated columns and rows:"
      f"{len(concatenated_restaurant_features)}")

# set index of data frame as place id
restaurants_ids = concatenated_restaurant_features['placeID']
concatenated_restaurant_features.index = restaurants_ids

# drop column userID and placeID as they do not impact on the rating
# drop columns food_rating and service_rating to see how restaurant features impact only general rating
columns_to_drop = ["placeID", "userID", "food_rating", "service_rating"]
concatenated_restaurant_features_and_general_rating = concatenated_restaurant_features
concatenated_restaurant_features_and_general_rating.drop(columns_to_drop, axis=1, inplace=True)

# write concatenated restaurant data frame to csv file
write_df_to_csv(data_dir="clean_data", file_name="concatenated_restaurant_features_and_general_rating.csv",
                data_frame=concatenated_restaurant_features_and_general_rating)

# split into input features (X) and output (y) variables to be predicted
input_features = concatenated_restaurant_features_and_general_rating.iloc[:, :-1].values
general_rating = concatenated_restaurant_features_and_general_rating.iloc[:, -1].values
input_features = input_features.astype(str)

# encoding input features
input_features = encode_input_features(input_features)
# encoding target labels
general_rating = encode_labels(general_rating)

# storing result as data frame
input_features_df = pd.DataFrame(input_features, index=restaurants_ids)
general_rating_df = pd.DataFrame(general_rating, index=restaurants_ids)

# storing result data frame to file
write_df_to_csv(data_dir="clean_data", file_name="encoded_input_features.csv",
                data_frame=input_features_df)
write_df_to_csv(data_dir="clean_data", file_name="encoded_labels.csv",
                data_frame=general_rating_df)
print(input_features_df.head())
# split into train and test sets
input_features_train, input_features_test, general_rating_train, general_rating_test = train_test_split(input_features,
                                                                                                        general_rating,
                                                                                                        test_size=0.33,
                                                                                                        random_state=1)

print('Train', input_features_train.shape, general_rating_train.shape)
print('Test', input_features_test.shape, general_rating_test.shape)

# categorical feature selection
input_train_fs, input_test_fs, selected_features = select_features(input_features_train, general_rating_train,
                                                                   input_features_test)
# what are scores for the features
for i in range(len(selected_features.scores_)):
    print('Feature %d: %f' % (i, selected_features.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(selected_features.scores_))], selected_features.scores_)
pyplot.show()

model = LogisticRegression(solver='lbfgs')
model.fit(input_features_train, general_rating_train)
# evaluate the model
general_rating_predictions = model.predict(input_features_test)
# evaluate predictions
accuracy = accuracy_score(general_rating_test, general_rating_predictions)
print('Accuracy: %.2f' % (accuracy * 100))
