import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def backwardElimination(x, y, sl):
    global regressor_OLS
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x

dataset = pd.read_csv('Multiple_Linear_Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
oneHotEncoder = OneHotEncoder(categories='auto')
columnTransformer = ColumnTransformer([('encoder', oneHotEncoder, [3])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X))
X = X[:, 1:]
X = np.append(arr=np.ones(shape=(50, 1)), values=X, axis=1)
print("X: \n%s" % X)

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_optimal_float = np.array(X_opt, dtype=float)

X_Modeled = backwardElimination(X_optimal_float, y, SL)