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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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


def select_features_chi_squared_method(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# feature selection
def select_features_mutual_info_classif(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k=5)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs

def select_relevant_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k=4)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs

joined_data = pd.read_csv('data/joined_data.csv')

# drop nan
joined_data = drop_nan(joined_data)
# see how many records we have after dropping NaN
print(f"Number of concatenated records after dropping NaN values:{len(joined_data)}")

# drop duplicated rows and columns
joined_data = drop_duplicated_rows_and_columns(joined_data)

# see how many records we have after dropping duplicated columns and rows
print(f"Number of records after dropping duplicated columns and rows:"
      f"{len(joined_data)}")

# set index of data frame as place id


# drop column userID and placeID as they do not impact on the rating
# drop columns food_rating and service_rating to see how restaurant features impact only general rating
columns_to_drop = ["placeID", "userID", "food_rating",
                   "service_rating", "longitude","latitude",
                   'the_geom_meter','name','address',
                   'city','state','country','zip','franchise','hours','days']
joined_data_copy = joined_data
joined_data_copy.drop(columns_to_drop, axis=1, inplace=True)
print(joined_data_copy.columns)
input_features = joined_data_copy.iloc[:, :-1].values
general_rating = joined_data_copy.iloc[:, -1].values
input_features = input_features.astype(str)

# encoding input features
input_features = encode_input_features(input_features)
# encoding target labels
general_rating = encode_labels(general_rating)

input_features_train, input_features_test, general_rating_train, general_rating_test = train_test_split(input_features,
                                                                                                        general_rating,
                                                                                                        test_size=0.33,
                                                                                                        random_state=1)

print('Train', input_features_train.shape, general_rating_train.shape)
print('Test', input_features_test.shape, general_rating_test.shape)

# categorical feature selection
input_train_fs, input_test_fs, selected_features = select_features_chi_squared_method(input_features_train, general_rating_train,
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

input_train_fs, input_test_fs = select_features_mutual_info_classif(input_features_train, general_rating_train,
                                                                   input_features_test)
model_for_selected_features = LogisticRegression(solver='lbfgs')
model_for_selected_features.fit(input_train_fs, general_rating_train)
# evaluate the model
general_rating_predictions_selected_features = model_for_selected_features.predict(input_test_fs)
# evaluate predictions
accuracy_selected_features = accuracy_score(general_rating_test, general_rating_predictions)
print('Accuracy: %.2f' % (accuracy_selected_features * 100))

columns_to_drop = ["dress_code", "accessibility", "Rambience",
                   "area", "other_services","Rpayment"]
joined_data_copy.drop(columns_to_drop, axis=1, inplace=True)
print(joined_data_copy.columns)
input_features = joined_data_copy.iloc[:, :-1].values
general_rating = joined_data_copy.iloc[:, -1].values
input_features = input_features.astype(str)

# encoding input features
input_features = encode_input_features(input_features)
# encoding target labels
general_rating = encode_labels(general_rating)

input_features_train, input_features_test, general_rating_train, general_rating_test = train_test_split(input_features,
                                                                                                        general_rating,
                                                                                                        test_size=0.33,
                                                                                                        random_state=1)

print('Train', input_features_train.shape, general_rating_train.shape)
print('Test', input_features_test.shape, general_rating_test.shape)


model = LogisticRegression(solver='lbfgs')
model.fit(input_features_train, general_rating_train)
# evaluate the model
general_rating_predictions = model.predict(input_features_test)
# evaluate predictions
accuracy = accuracy_score(general_rating_test, general_rating_predictions)
print('Accuracy: %.2f' % (accuracy * 100))
