import pandas as pd
import numpy as np
from itertools import islice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from math import isnan


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


def take(n, iterable):
    return list(islice(iterable, n))


def find_selected_features_with_chi_squared_method(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    fs.transform(X_train)
    fs.transform(X_test)
    features_scores = {}
    for i in range(len(fs.scores_)):
        features_scores[i] = fs.scores_[i]
    return fs, features_scores


def find_selected_features_with_mutual_information_stats(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=5)
    fs.fit(X_train, y_train)
    fs.transform(X_train)
    fs.transform(X_test)
    features_scores = {}
    for i in range(len(fs.scores_)):
        features_scores[i] = fs.scores_[i]
    return fs, feature_scores


def visualise_selected_features(selected_features):
    for i in range(len(selected_features.scores_)):
        print('Feature %d: %f' % (i, selected_features.scores_[i]))
    pyplot.bar([i for i in range(len(selected_features.scores_))], selected_features.scores_)
    pyplot.show()


def filter_selected_features(data_frame, features_scores, n):
    data_frame_copy = data_frame
    clean_features_scores = {k: features_scores[k] for k in features_scores if not isnan(features_scores[k])}
    feature_scores_sorted_desc = {key: value for key, value in
                                  sorted(clean_features_scores.items(), reverse=True, key=lambda item: item[1])}
    selected_features = take(n, feature_scores_sorted_desc.items())
    selected_features_columns = [item[0] for item in selected_features]
    columns_to_drop = list(
        filter(lambda element: element not in selected_features_columns, np.array(data_frame_copy.columns)))
    data_frame_copy.drop(columns_to_drop, axis=1, inplace=True)
    return data_frame_copy


def calculate_selected_features_prediction_accuracy(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)
    # evaluate the model
    y_predictions = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_predictions)

    return accuracy

# encoding input features
input_features = [[]]
# encoding target labels
general_rating = []

X_train_all_features, X_test_all_features, y_train_all_features, y_test_all_features \
    = train_test_split(input_features, general_rating, test_size=0.33, random_state=1)

accuracy = calculate_selected_features_prediction_accuracy(X_train_all_features, y_train_all_features, X_test_all_features, y_test_all_features)
print('Accuracy: %.2f' % (accuracy * 100))

# Finding selected features with chi-squared method
selected_features, feature_scores = find_selected_features_with_chi_squared_method(X_train_all_features,
                                                                                   y_train_all_features,
                                                                                   X_test_all_features)
visualise_selected_features(selected_features)
input_selected_features = filter_selected_features(pd.DataFrame(input_features), feature_scores, 5)
print(input_selected_features.columns)

X_train_selected_features, X_test_selected_features, y_train_selected_features, y_test_selected_features \
    = train_test_split(input_selected_features,general_rating,test_size=0.33, random_state=1)

accuracy = calculate_selected_features_prediction_accuracy(X_train_selected_features, y_train_selected_features, X_test_selected_features, y_test_selected_features)
print('Accuracy: %.2f' % (accuracy * 100))

# Finding selected features with chi-squared method
selected_features, feature_scores = find_selected_features_with_mutual_information_stats(X_train_all_features,
                                                                                         y_train_all_features,
                                                                                         X_test_all_features)
visualise_selected_features(selected_features)
input_selected_features = filter_selected_features(pd.DataFrame(input_features), feature_scores, 5)
print(input_selected_features.columns)

X_train_selected_features, X_test_selected_features, y_train_selected_features, y_test_selected_features \
    = train_test_split(input_selected_features,general_rating,test_size=0.33, random_state=1)

accuracy = calculate_selected_features_prediction_accuracy(X_train_selected_features, y_train_selected_features, X_test_selected_features, y_test_selected_features)
print('Accuracy: %.2f' % (accuracy * 100))


