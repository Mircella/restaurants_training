import pandas as pd
import numpy as np
from itertools import islice
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from math import isnan
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    fs.transform(X_train)
    fs.transform(X_test)
    features_scores = {}
    for i in range(len(fs.scores_)):
        features_scores[i] = fs.scores_[i]
    return fs, features_scores


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


def calculate_selected_features_prediction_accuracy_with_logistic_regression(X_train, y_train, X_test, y_test):
    logistic_regression_classifier = LogisticRegression(solver='lbfgs')
    logistic_regression_classifier.fit(X_train, y_train)
    # evaluate the model
    y_predictions = logistic_regression_classifier.predict(X_test)
    # evaluate predictions

    return y_predictions


def calculate_selected_features_prediction_accuracy_with_random_forest_classifier(X_train, y_train, X_test, y_test):
    random_forest_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    random_forest_classifier.fit(X_train, y_train)
    # evaluate the model
    y_predictions = random_forest_classifier.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_predictions)

    return accuracy


def calculate_selected_features_prediction_accuracy_with_random_forest_regression(X_train, y_train, X_test, y_test):
    # max_features(default 'auto') - number of features to consider when looking for the best split
    random_forest_regressor = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=0)
    random_forest_regressor.fit(X_train, y_train)
    y_prediction = random_forest_regressor.predict(X_test)

    return y_prediction

def calculate_selected_features_prediction_accuracy_with_multiple_linear_regression(X_train, y_train, X_test, y_test):
    # max_features(default 'auto') - number of features to consider when looking for the best split
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    return y_pred