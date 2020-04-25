import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


pj = pd.read_csv('userprofile.csv', delimiter = ';')
pj = pj.replace('?', np.NaN)
pj_clean = pj.dropna()
pj_clean_obj = pj_clean.iloc[:,[0,3,4,5,6,7,8,9,10,11,12,13,14,17]].values


for i in range(1,8):
    labelencoder_X = LabelEncoder()
    pj_clean_obj[:, i] = labelencoder_X.fit_transform(pj_clean_obj[:, i])

for i in range(9,14):
    labelencoder_X = LabelEncoder()
    pj_clean_obj[:, i] = labelencoder_X.fit_transform(pj_clean_obj[:, i])
    
pj_new = pd.DataFrame(pj_clean_obj[:,:])

pj_cluster = pj_clean_obj[:,1:]

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init
                    = 10, random_state = 0)
    kmeans.fit(pj_cluster)
    wcss.append(kmeans.inertia_) #inertia_ computes wcss
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10,
                random_state = 0)
y_kmeans = kmeans.fit_predict(pj_cluster)




















from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OrdinalEncoder
def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

pj_col = ['smoker']
for columns in pj_col:
    encode(impute_data(columns))

'''dataset['trackable_name'].unique()
df_new = dataset.iloc[:,6:].copy()
df_unique = df_new.drop_duplicates()
df.C.nunique(dropna = True)''' 


df_unique['trackable_type'].unique()
df_symptom = df_unique.loc[df_unique['trackable_type'] == 'Symptom']

geo_df = pd.read_csv('geoplaces2.csv', delimiter = ';', encoding='latin-1')

pivot = pd.pivot_table(pj, values = ['userID'],
                       index = 'budget', columns = ['activity'],
                       aggfunc = 'count')

print(pivot)

X_pj = pj.iloc[:, [6,8,14,17]]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_pj[:, 0])
X_pj[:, 0] = imputer.transform(X_pj[:, 0])


