import pandas as pd
import numpy as np
from utils.functions_for_encoding import drop_columns
from utils.functions_for_encoding import label_encoding

df = pd.read_csv('data/whole_rest_data.csv')

df = drop_columns(df,'latitude', 'longitude', 'the_geom_meter',
                     'name','address','city','state','country','zip','fax', 'url',
                     'franchise','hours','days','userID','placeID')
print(df.columns)
X = df.iloc[:,:-3].values
y = df.iloc[:,-3].values
food_rating = df.iloc[:,-2].values
service_rating = df.iloc[:,-1].values

X_serv = df.iloc[:,[0,1,3,4,6,7,9,10]].values

input_features = label_encoding(X_serv,8)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_serv, service_rating, test_size = 0.2, random_state = 0)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators = 200)
# n_estimators = 100 (default)
# loss function = deviance(default) used in Logistic Regression
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
clf.score(X_test, y_test)


# XGBoost 
from xgboost import XGBClassifier
clf = XGBClassifier()
# n_estimators = 100 (default)
# max_depth = 3 (default)
clf.fit(x_train,y_train)
clf.predict(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
comparison = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis = 1)

X = np.append(arr = np.ones((6276,1)).astype(int), values = X, axis = 1)


X_opt = X[:, [0,1,2,3]]

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()