import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#dataset = pd.read_csv('data\joined_data.csv')
#clean = dataset.drop(['the_geom_meter','name', 'address','city','state','country',
#                      'zip','Rambience','franchise','other_services','userID'], axis = 1)
#irrelevant as only informal and casual
#clean = clean.drop('dress_code', axis = 1)
#cat_clean = clean.select_dtypes(include=['object']).copy() #independent X

concat = pd.read_csv('../data/joined_data.csv')
concat_dep = concat.drop(['placeID','the_geom_meter','name', 'address','city','state','country',
                      'zip','latitude','longitude','franchise','hours',
                      'days','userID'], axis = 1) #!if concat data, drop fax and url

y = concat_dep.iloc[:,-3].values #dependent; rating
concat_dep_obj = concat_dep.iloc[:,:-3].values

for i in range(0,11): #if concat data, increase i
    labelencoder_X = LabelEncoder()
    concat_dep_obj[:, i] = labelencoder_X.fit_transform(concat_dep_obj[:, i])

#onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13])
#X = onehotencoder.fit_transform(concat_dep_obj).toarray()

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(concat_dep_obj, y, test_size = 0.2, random_state = 0)

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

