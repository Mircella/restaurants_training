import pandas as pd
from utils.functions_for_encoding import drop_columns
from utils.functions_for_encoding import label_encoding
from sklearn.model_selection import train_test_split
from feature_selection_ratings.feature_selection import find_selected_features_with_chi_squared_method
from feature_selection_ratings.feature_selection import find_selected_features_with_mutual_information_stats
from feature_selection_ratings.feature_selection import visualise_selected_features
from feature_selection_ratings.feature_selection import filter_selected_features
from feature_selection_ratings.feature_selection import calculate_selected_features_prediction_accuracy_with_random_forest_regression
from feature_selection_ratings.feature_selection import calculate_selected_features_prediction_accuracy_with_logistic_regression
from feature_selection_ratings.feature_selection import calculate_selected_features_prediction_accuracy_with_multiple_linear_regression
from sklearn.metrics import r2_score

encoded_restaurant_data = pd.read_csv('clean_data_encoded/all_restaurant_data_encoded.csv')

print(encoded_restaurant_data.columns)
X = encoded_restaurant_data.iloc[:,2:].values
y = encoded_restaurant_data.iloc[:, 1].values


# General rating
X_train_all_features, X_test_all_features, y_train_all_features, y_test_all_features \
    = train_test_split(X, y, test_size=0.33, random_state=1)

predictions = calculate_selected_features_prediction_accuracy_with_multiple_linear_regression(X_train_all_features, y_train_all_features, X_test_all_features, y_test_all_features)
predictions_vs_real_1 = []
for i, prediction in enumerate(predictions):
    predictions_vs_real_1.append((prediction, y_test_all_features[i]))
    
r2_score(y_test_all_features,predictions)

# Finding selected features with chi-squared method
selected_features, feature_scores = find_selected_features_with_chi_squared_method(X_train_all_features,
                                                                                   y_train_all_features,
                                                                                   X_test_all_features)
visualise_selected_features(selected_features)
input_selected_features = filter_selected_features(pd.DataFrame(X), feature_scores, 5)
print(input_selected_features.columns)

X_train_selected_features, X_test_selected_features, y_train_selected_features, y_test_selected_features \
    = train_test_split(input_selected_features, y, test_size=0.33, random_state=1)

predictions = calculate_selected_features_prediction_accuracy_with_multiple_linear_regression(X_train_selected_features, y_train_selected_features, X_test_selected_features, y_test_selected_features)
predictions_vs_real_2 = []
for i, prediction in enumerate(predictions):
    predictions_vs_real_2.append((prediction, y_test_all_features[i]))

# Finding selected features with mutual_information_stats method
selected_features, feature_scores = find_selected_features_with_mutual_information_stats(X_train_all_features,
                                                                                         y_train_all_features,
                                                                                         X_test_all_features)
visualise_selected_features(selected_features)
input_selected_features = filter_selected_features(pd.DataFrame(X), feature_scores, 5)
print(input_selected_features.columns)

X_train_selected_features, X_test_selected_features, y_train_selected_features, y_test_selected_features \
    = train_test_split(input_selected_features, y, test_size=0.33, random_state=1)

predictions = calculate_selected_features_prediction_accuracy_with_multiple_linear_regression(X_train_selected_features, y_train_selected_features, X_test_selected_features, y_test_selected_features)
predictions_vs_real_3 = []
for i, prediction in enumerate(predictions):
    predictions_vs_real_3.append((prediction, y_test_all_features[i]))
print("")