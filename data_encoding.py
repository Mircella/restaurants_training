import pandas as pd
import numpy as np
from func_for_enc import drop_columns
from func_for_enc import label_encoding
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

joined_restaurant_data = pd.read_csv('data\whole_rest_data.csv')

df  = drop_columns(joined_restaurant_data,'latitude', 'longitude', 'the_geom_meter',
                     'name','address','city','state','country','fax','zip','url',
                     'franchise','hours','days','userID')

X = df.iloc[:,:-3].values
X = X[:,1:]
y_rating = df.iloc[:,-3].values
y_food_rating = df.iloc[:,-2].values
y_service_rating = df.iloc[:,-1].values

X = label_encoding(X,11)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_food_rating, test_size = 0.2, random_state = 0)

def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

input_train_fs, input_test_fs, selected_features = select_features(X_train, y_train, X_test)
for i in range(len(selected_features.scores_)):
	print('Feature %d: %f' % (i, selected_features.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(selected_features.scores_))], selected_features.scores_)
pyplot.show()

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
# evaluate the model
general_rating_predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, general_rating_predictions)
print('Accuracy: %.2f' % (accuracy*100))
