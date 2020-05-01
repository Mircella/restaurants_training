#rating average
import pandas as pd
from utils.functions_for_encoding import drop_columns
from utils.utils_for_files_storing_and_reading import write_df_to_csv


# #rest data
# r_payment = pd.read_csv('../data/chefmozaccepts.csv', delimiter = ';')
# r_cuisine = pd.read_csv('../data/chefmozcuisine.csv', delimiter = ';')
# r_hours = pd.read_csv('../data/chefmozhours.csv', delimiter = ',')
# r_parking = pd.read_csv('../data/chefmozparking.csv', delimiter = ';')
# r_general = pd.read_csv('../data/geoplaces.csv', delimiter = ';', encoding='latin-1')


#rest data
r_payment = pd.read_csv('../clean_data/restaurant_payment_types.csv')
r_cuisine = pd.read_csv('../clean_data/restaurant_cuisine_types.csv')
r_hours = pd.read_csv('../clean_data/restaurant_working_hours.csv')
r_parking = pd.read_csv('../clean_data/restaurant_parking_types.csv')
r_general = pd.read_csv('../clean_data/restaurant_geo_places.csv')

def lowercase(df, *columns):
    for column in columns:
        df[column] = df[column].str.lower()

def unique_values(df, *columns):
    for column in columns:
        return pd.DataFrame(df[column].unique())

#cuisine
lowercase(r_cuisine, 'Rcuisine')
unique_cuisine = unique_values(r_cuisine, 'Rcuisine')
cuisine_pivot = pd.pivot_table(r_cuisine, values = 'Rcuisine', index = 'Rcuisine', aggfunc = 'count')

#parking
lowercase(r_parking, 'parking_lot')
unique_parking = unique_values(r_parking, 'parking_lot')


#payment
lowercase(r_payment, 'Rpayment')
unique_payment = unique_values(r_payment, 'Rpayment')

#general
r_general = drop_columns(r_general, 'latitude', 'longitude', 'the_geom_meter', 'name', 'address',
             'city','state','country','fax','zip','url','franchise')
lowercase(r_general, 'alcohol', 'smoking_area', 'dress_code', 'accessibility',
          'price', 'Rambience', 'area', 'other_services')

unique_alcohol = unique_values(r_general, 'alcohol')

unique_smoking = unique_values(r_general, 'smoking_area')
r_general['smoking_area']=r_general['smoking_area'].replace(['none'],'not permitted')
r_general['smoking_area']=r_general['smoking_area'].replace(['only at bar', 
         'section'],'permitted')

unique_dress_code = unique_values(r_general, 'dress_code')
r_general['dress_code']=r_general['dress_code'].replace(['casual'],'informal')

unique_accessibility = unique_values(r_general, 'accessibility')
unique_price = unique_values(r_general, 'price')
unique_Rambience = unique_values(r_general, 'Rambience')
unique_area = unique_values(r_general, 'area')
unique_other_services = unique_values(r_general, 'other_services')

#hours was not changed

#df to csv
write_df_to_csv('no_similiar_categories', 'restaurant_cuisine.csv', r_cuisine)
write_df_to_csv('no_similiar_categories', 'restaurant_general.csv', r_general)
write_df_to_csv('no_similiar_categories', 'restaurant_hours.csv', r_hours)
write_df_to_csv('no_similiar_categories', 'restaurant_parking.csv', r_parking)
write_df_to_csv('no_similiar_categories', 'restaurant_payment.csv', r_payment)

#test = df_r.groupby('placeID')['rating'].mean() - for mean
#rating_pivot = pd.pivot_table(ratings, values = ['rating', 'food_rating', 'service_rating'], index = 'placeID', aggfunc = 'mean')