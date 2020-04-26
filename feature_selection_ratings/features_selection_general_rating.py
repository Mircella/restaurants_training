import pandas as pd
from utils.functions_for_encoding import drop_columns
from utils.functions_for_encoding import label_encoding
from sklearn.model_selection import train_test_split
from feature_selection_ratings.feature_selection import find_selected_features_with_chi_squared_method
from feature_selection_ratings.feature_selection import find_selected_features_with_mutual_information_stats
from feature_selection_ratings.feature_selection import visualise_selected_features
from feature_selection_ratings.feature_selection import filter_selected_features
from feature_selection_ratings.feature_selection import calculate_selected_features_prediction_accuracy_with_random_forest_classifier


joined_restaurant_data = pd.read_csv('../data/whole_rest_data.csv', encoding='utf-8')

df = drop_columns(joined_restaurant_data,'latitude', 'longitude', 'the_geom_meter',
                     'name','address','city','state','country','zip','fax','url',
                     'franchise','hours','days','userID','placeID')
print(df.columns)
X = df.iloc[:,:-3].values
general_rating = df.iloc[:,-3].values
food_rating = df.iloc[:,-2].values
service_rating = df.iloc[:,-1].values

input_features = label_encoding(X,11)

# General rating
X_train_all_features, X_test_all_features, y_train_all_features, y_test_all_features \
    = train_test_split(input_features, general_rating, test_size=0.33, random_state=1)

accuracy = calculate_selected_features_prediction_accuracy_with_random_forest_classifier(X_train_all_features, y_train_all_features, X_test_all_features, y_test_all_features)
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

accuracy = calculate_selected_features_prediction_accuracy_with_random_forest_classifier(X_train_selected_features, y_train_selected_features, X_test_selected_features, y_test_selected_features)
print('Accuracy: %.2f' % (accuracy * 100))

# Finding selected features with mutual_information_stats method
selected_features, feature_scores = find_selected_features_with_mutual_information_stats(X_train_all_features,
                                                                                         y_train_all_features,
                                                                                         X_test_all_features)
visualise_selected_features(selected_features)
input_selected_features = filter_selected_features(pd.DataFrame(input_features), feature_scores, 5)
print(input_selected_features.columns)

X_train_selected_features, X_test_selected_features, y_train_selected_features, y_test_selected_features \
    = train_test_split(input_selected_features,general_rating,test_size=0.33, random_state=1)

accuracy = calculate_selected_features_prediction_accuracy_with_random_forest_classifier(X_train_selected_features, y_train_selected_features, X_test_selected_features, y_test_selected_features)
print('Accuracy: %.2f' % (accuracy * 100))
