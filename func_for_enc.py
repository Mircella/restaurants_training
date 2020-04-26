import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def drop_columns(df, *columns):
    df_dropped = df.drop([column for column in columns], axis = 1)
    return df_dropped
   
def label_encoding(df,num_of_col):
    for i in range(0,num_of_col): 
        labelencoder_X = LabelEncoder()
        df[:, i] = labelencoder_X.fit_transform(df[:, i])
    return df
    
