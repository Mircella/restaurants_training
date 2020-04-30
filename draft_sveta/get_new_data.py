#rating average
import pandas as pd
from utils.functions_for_encoding import drop_columns
from utils.utils_for_files_storing_and_reading import write_df_to_csv


#rest data
r_payment = pd.read_csv('data/chefmozaccepts.csv', delimiter = ';')
r_cuisine = pd.read_csv('data/chefmozcuisine.csv', delimiter = ';')
r_hours = pd.read_csv('data/chefmozhours.csv', delimiter = ',')
r_parking = pd.read_csv('data/chefmozparking.csv', delimiter = ';')
r_general = pd.read_csv('data/geoplaces.csv', delimiter = ';', encoding='latin-1')


def lowercase(df, *columns):
    for column in columns:
        df[column] = df[column].str.lower()

def unique_values(df, *columns):
    for column in columns:
        return pd.DataFrame(df[column].unique())
    
r_general = drop_columns(r_general, 'latitude', 'longitude', 'the_geom_meter', 'name', 'address',
             'city','state','country','fax','zip','url','franchise')
lowercase(r_general, 'alcohol', 'smoking_area', 'dress_code', 'accessibility',
          'price', 'Rambience', 'area', 'other_services')

test = unique_values(r_parking, 'parking_lot')

r_general['smoking_area']=r_general['smoking_area'].replace(['none'],'not permitted')
r_general['smoking_area']=r_general['smoking_area'].replace(['only at bar', 
         'section'],'permitted')
r_general['dress_code']=r_general['dress_code'].replace(['casual'],'informal')


#df to csv
write_df_to_csv('new_data','restaurant_cuisine',r_cuisine)
write_df_to_csv('new_data','restaurant_geo',r_general)
write_df_to_csv('new_data','restaurant_hours',r_hours)
write_df_to_csv('new_data','restaurant_parking',r_parking)
write_df_to_csv('new_data','restaurant_payment',r_payment)

#test = df_r.groupby('placeID')['rating'].mean() - for mean