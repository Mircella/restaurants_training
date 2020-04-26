import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def drop_columns(df, *columns):
    df_dropped = df.drop([column for column in columns], axis = 1)
    return df_dropped
   
    
    
    
    
    
concat = pd.read_csv('data\joined_data.csv')
concat_dep = concat.drop(['placeID','the_geom_meter','name', 'address','city','state','country',
                      'zip','latitude','longitude','franchise','hours',
                      'days','userID'], axis = 1)
    
test  = drop_columns(concat, 'placeID','name')