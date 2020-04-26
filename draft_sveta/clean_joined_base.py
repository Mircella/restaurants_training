#all from clean_joined

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('../data/joined_data.csv')
clean = dataset.drop(['the_geom_meter','name', 'address','city','state','country',
                      'zip','Rambience','franchise','other_services','userID'], axis = 1)

#irrelevant as only informal and casual
clean = clean.drop('dress_code', axis = 1)

cat_clean = clean.select_dtypes(include=['object']).copy() #independent X
y = clean.iloc[:,-3].values #dependent rating

cat_clean_obj = cat_clean.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for i in range(0,10):
    labelencoder_X = LabelEncoder()
    cat_clean_obj[:, i] = labelencoder_X.fit_transform(cat_clean_obj[:, i])

onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7,8,9])
test = onehotencoder.fit_transform(cat_clean_obj).toarray()


#label_encoding another approach - mb redundant
'''cat_clean_copy = cat_clean.copy()
enc_columns = ['alcohol', 'smoking_area', 'accessibility', 'price', 'area', 
               'Rcuisine', 'hours', 'days', 'parking_lot', 'Rpayment']
for column in enc_columns:
    cat_clean_copy[column] = cat_clean_copy[column].astype('category')
    cat_clean_copy[column] = cat_clean_copy[column].cat.codes'''

#test - independent X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(test, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor_lin = LinearRegression()
regressor_lin.fit(X_train, y_train)

y_pred = regressor_lin.predict(X_test)
regressor_lin.score(X_test,y_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


clf = RandomForestClassifier(max_depth = 40,  n_estimators =200)                 
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

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





#the below is some useful stuff

print(cat_dataset['userID'].value_counts())
print(cat_dataset['userID'].value_counts().count())

%matplotlib inline
import seaborn as sns
rest_count = cat_dataset['city'].value_counts()
sns.set(style="darkgrid")
sns.barplot(rest_count.index, rest_count.values, alpha=0.9)
plt.title('Frequency Distribution of City')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.show()

labels = cat_dataset['city'].astype('category').cat.categories.tolist()
counts = cat_dataset['city'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


values, counts = np.unique(dataset['city'], return_counts=True)
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

dataset.dtypes

dataset.apply(dataset['city'].str.lower())
xSecureLower['placeID'].unique()


xSecureLower = dataset.applymap(lambda x: x.lower() if isinstance(x,str) else x)
xSecureLower.dtypes

import re
#first step: keep only letters. punctuation and numbers will be removed
review = re.sub('[^a-zA-Z0-9]','', dataset['city'][0])
review = re.sub('[^a-zA-Z0-9]','', dataset['city'])


xSecureLower2 = dataset.applymap(lambda x: re.sub('[^a-zA-Z0-9]','',dataset['city'][x]) if isinstance(x,str) else x)