#ols backward elimination
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('clean_data_encoded/all_restaurant_data_encoded.csv')
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

independent_variables = [(element , 1) for element in np.asarray(df.columns)[2:97]]

X = X[:, 1:]
X = np.append(arr=np.ones(shape=(130, 1)), values=X, axis=1)

def backwardElimination(x, y, sl, independent_variables):
    global regressor_OLS
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            temp_0 = independent_variables[i][0]
            independent_variables[i] = (temp_0, 0)
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
              25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,
              47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,
              70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94]]

X_Modeled = backwardElimination(X_opt, y, SL, independent_variables)

selected_features = []
for x in range(0, len(independent_variables)):
    if independent_variables[x][1] == 1:
        selected_features.append(independent_variables[x][0])

print('Features that affect the rating most are:')
for feature in selected_features:
    print(f'-{feature}')

    