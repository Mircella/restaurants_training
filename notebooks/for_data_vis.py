import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
from utils.data_frames_cleaning_functions import join_tables
from utils.data_frames_cleaning_functions import concatenate_tables
from utils.data_frames_cleaning_functions import drop_duplicated_rows_and_columns
from utils.data_frames_cleaning_functions import find_unique_records_number_by_column
from utils.functions_for_encoding import drop_columns
from utils.data_frames_cleaning_functions import drop_nan
from utils.utils_for_files_storing_and_reading import write_df_to_csv

# loading restaurants data
restaurant_payment_types = pd.read_csv('data/chefmozaccepts.csv', delimiter =';')
restaurant_cuisine_types = pd.read_csv('data/chefmozcuisine.csv', delimiter =';')
restaurant_working_hours = pd.read_csv('data/chefmozhours.csv', delimiter =',')
restaurant_parking = pd.read_csv('data/chefmozparking.csv', delimiter =';')
restaurant_geo_places = pd.read_csv('data/geoplaces.csv', delimiter =';', encoding='latin-1')
ratings = pd.read_csv('data/rating_final.csv', delimiter =';')


# I created new directory with clean .csv files that have unified formatting as input data have different delimiters
write_df_to_csv('clean_data', 'restaurant_payment_types.csv', restaurant_payment_types)
write_df_to_csv('clean_data', 'restaurant_cuisine_types.csv', restaurant_cuisine_types)

# as restaurant_working_hours data set has mixed types of delimiters, I rewrite its columns 'hours' and 'days' to be consistent in formatting with the rest of data sets
for i, series in restaurant_working_hours.iterrows():
    hours = restaurant_working_hours.loc[i, 'hours'][0:len(restaurant_working_hours.loc[i, 'hours']) - 1]
    restaurant_working_hours.loc[i, 'hours'] = hours
    hours = restaurant_working_hours.loc[i, 'days'][0:len(restaurant_working_hours.loc[i, 'days']) - 1]
    restaurant_working_hours.loc[i, 'days'] = hours

write_df_to_csv('clean_data', 'restaurant_working_hours.csv', restaurant_working_hours)
write_df_to_csv('clean_data', 'restaurant_parking_types.csv', restaurant_parking)
write_df_to_csv('clean_data', 'restaurant_geo_places.csv', restaurant_geo_places)
write_df_to_csv('clean_data', 'restaurant_ratings.csv', ratings)


#!!!to add
#As I wrote all restaurant related data into correct format, I will use it for the further data analysis.
r_payment = pd.read_csv('../clean_data/restaurant_payment_types.csv')
r_cuisine = pd.read_csv('../clean_data/restaurant_cuisine_types.csv')
r_hours = pd.read_csv('../clean_data/restaurant_working_hours.csv')
r_parking = pd.read_csv('../clean_data/restaurant_parking_types.csv')
r_general = pd.read_csv('../clean_data/restaurant_geo_places.csv')


#!!!to add
#Some rows in the data have the same value but written in a different format, e.g. 'VISA' vs 'Visa'. 
r_payment['Rpayment'].unique()

#I will use the below functions to correct the data.
def lowercase(df, *columns):
    for column in columns:
        df[column] = df[column].str.lower()

def unique_values(df, *columns):
    for column in columns:
        return pd.DataFrame(df[column].unique())
    
#The cuisine data
lowercase(r_cuisine, 'Rcuisine')
unique_cuisine = unique_values(r_cuisine, 'Rcuisine')
#There are no duplictes or some values which can be merged so I leave it like it is.
r_cuisine.Rcuisine.value_counts()
# Extracting how many cuisine types exist in restaurants
print(f"Number of unique restaurants with cuisine type specified:{len(r_cuisine['placeID'].unique())}")
print(f"Number of cuisine types:{len(r_cuisine['Rcuisine'].unique())}")
#The barplot for cuisine. As there are 59 different types of cuisine, I will display top 10 by frequency.  
cuisine_count = r_cuisine['Rcuisine'].value_counts()
sns.set(style="darkgrid")
sns.barplot(cuisine_count.index[:10], cuisine_count.values[:10], alpha=0.9)
plt.title('Top 10 Frequent Cuisines')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Cuisine', fontsize=12)
plt.show()


#The parking data
lowercase(r_parking, 'parking_lot')
unique_parking = unique_values(r_parking, 'parking_lot')
#There are no duplictes or some values which can be merged so I leave it like it is.
r_parking.parking_lot.value_counts()
#The barplot for parking
parking_count = r_parking['parking_lot'].value_counts()
sns.set(style="darkgrid")
sns.barplot(parking_count.index, parking_count.values, alpha=0.9)
plt.title('Parking Options')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Parking', fontsize=12)
plt.show()


#The payment data
lowercase(r_payment, 'Rpayment')
unique_payment = unique_values(r_payment, 'Rpayment')
#There are no duplictes or some values which can be merged so I leave it like it is.
r_payment.Rpayment.value_counts()
#The barplot for payment
payment_count = r_payment['Rpayment'].value_counts()
sns.set(style="darkgrid")
sns.barplot(payment_count.index, payment_count.values, alpha=0.9)
plt.title('Payment Options')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Payment', fontsize=12)
plt.show()

#The working schedule data
lowercase(r_hours, 'days')
r_hours['hours'].value_counts()
pd.pivot_table(r_hours, values = ['hours'], index = 'hours', columns = 'days',aggfunc = 'count')
#There are 273 different shifts over three type of working days. As there are many different options of
#working hours of restaurants, it's been decided noto to consider this variable. If a client gave a rating to
#a restaurant, it assumes he attended it and the working shift is not an issue for the client. Moreover, 
#the data does not state what hours are more profitable for a restaurant and it's not clear how it can be evaluated. 


#The general data
#Some columns in the r_general will be removed. Those columns, in my opinion, will not affect the 
#rating and/or the visualization of those columns is irrelevant and does not shed a light on the data.
r_general = drop_columns(r_general, 'latitude', 'longitude', 'the_geom_meter', 'address',
             'fax','zip','url', 'name')
lowercase(r_general, 'city', 'state', 'country', 'alcohol', 'smoking_area', 
          'dress_code', 'accessibility', 'price', 'Rambience', 'franchise', 'area', 
          'other_services')

r_general['country'].value_counts()
#Mexico is a country for the mojrity of restaurants. Some restaurants with a missing
#country still have state and/or city indicatiting it's still Mexico. A few records 
#have neither country nor state nor city. Nevertheless, I assume the survey was
#conducted for Mexican restaurants. As this value is the same for all places, the
#column 'country' will be dropped from the dataset.
r_general = drop_columns(r_general,'country')


r_general['state'].value_counts()
#When checking state from the r_general, some values caught my attention. It's slp, 
#san luis potosi, s.l.p., san luis potos. They all refer to one state San Luis Potose. 
#I decided to replcase all those values by 'san luis potosi'. '?' were replaced by 'Nan'

r_general['state'] = r_general['state'].replace(['slp', 'san luis potosi',
         's.l.p.', 'san luis potos'],'san luis potosi')
r_general['state'] = r_general['state'].replace('?', 'Nan')

r_general['city'].value_counts()
#Some city values can be merged like it was done for the 'state'. '?' were replaced by 'Nan'
r_general['city'] = r_general['city'].replace(['san luis potosi', 'slp',
         's.l.p.', 's.l.p', 'san luis potos','san luis potosi '],'san luis potosi')
r_general['city'] = r_general['city'].replace(['victoria', 'victoria ',
         'ciudad victoria', 'cd victoria', 'cd. victoria'],'ciudad victoria')
r_general['city'] = r_general['city'].replace('?', 'Nan')
    
r_general['alcohol'].value_counts()  

r_general['smoking_area'].value_counts()
#I checked what options are in the smoking_area column. I deciced to have two categories:
#1)not permitted (none, not permitted) and 2) permitted(only at bar, section, permitted)
r_general['smoking_area'] = r_general['smoking_area'].replace(['none'],'not permitted')
r_general['smoking_area'] = r_general['smoking_area'].replace(['only at bar', 
         'section'],'permitted')

r_general['dress_code'].value_counts()
#I decided to unite informal and casual and eventually have only two options for
#dress_code: informal and formal.
r_general['dress_code'] = r_general['dress_code'].replace(['casual'],'informal')

#The values of rest columns from r_general. After checking the values, I decided
#to leave them like they are without any replacement.
print(r_general['accessibility'].value_counts())
print(r_general['price'].value_counts())
print(r_general['Rambience'].value_counts())
print(r_general['franchise'].value_counts())
print(r_general['area'].value_counts())
print(r_general['other_services'].value_counts())

#Barplots for r_general. As there are many categories in this dataset, I decided
#to write the general function to get the plot for "X' column fron r_general.
def plot_general(column):
    count = r_general[column].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(count.index, count.values, alpha=0.9)
    plt.title(f'{column.title()} Options')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(f'{column.title()}', fontsize=12)
    plt.show()    

plot_general('city')
plot_general('state')
plot_general('price')


#I saved the clean data sets + rating dataset (to have all sets in one directory)
#to csv files which will be used later.
write_df_to_csv('for_data_visualization', 'restaurant_cuisine.csv', r_cuisine)
write_df_to_csv('for_data_visualization', 'restaurant_general.csv', r_general)
write_df_to_csv('for_data_visualization', 'restaurant_hours.csv', r_hours)
write_df_to_csv('for_data_visualization', 'restaurant_parking.csv', r_parking)
write_df_to_csv('for_data_visualization', 'restaurant_payment.csv', r_payment)
write_df_to_csv('for_data_visualization', 'ratings.csv', ratings)


#rating dataset
#I decided to check the correlation between different rating from ratings dataset. I created a new dataframe
#to see the average rating of each rating grouped by placeID. I got a table of 130 restaurants with their ratings.
rating_mean = ratings.groupby('placeID')['rating', 'food_rating', 'service_rating'].mean().reset_index()
rating_mean.iloc[:,1:].describe()
correlation = rating_mean.iloc[:,1:].corr()
sns.heatmap(correlation)
#I assume rating and service_rating will be affected by the same variables.



#from jenya original
# Here I want to merge data of restaurants from different data sets by placeID with ratings and see how different features make an impactt on rating
#I will contonue to work with the clean datasets.  As I saved clean data sets, I can use those sets any time I need it.
from utils.data_frames_cleaning_functions import merge_and_group
# Merging rating and restaurant parking types to see how presence of parking can potentially impact the service or general rating

merge_and_group(
    left_df = ratings, 
    right_df= r_parking, 
    merge_column='placeID', 
    group_column='parking_lot', 
    estimate_column='service_rating'
)

parking_rating = pd.merge(rating_mean,r_parking,on = 'placeID', how = 'left')
subjects = ['rating', 'food_rating', 'service_rating']
dataset = parking_rating.groupby('parking_lot')[subjects].mean()

indx = np.arange(len(subjects))
score_label = np.arange(0,2,0.5)
print(dataset.T)
none_means = list(dataset.T['none'])
public_means = list(dataset.T['public'])

bar_width = 0.35

#create plot
fig, ax = plt.subplots()
barNone = ax.bar(indx - bar_width/2, none_means, bar_width, label = 'None means')
barPublic = ax.bar(indx + bar_width/2, public_means, bar_width, label = 'Public means')

#inserting x axis label
ax.set_xticks(indx)
ax.set_xticklabels(subjects)

#inserting y axis label
ax.set_yticks(score_label)
ax.set_yticklabels(score_label)

#inserting legend
ax.legend()

for i in indx:
    ax.annotate('{0:.0f}'.format(barNone[i].get_height()),
                xy=(barNone[i].get_x() + barNone[i].get_width()/2, barNone[i].get_height()),
                xytext = (0,3),
                ha = 'center',
                va = 'bottom'
                )

for i in indx:
    ax.annotate('{0:.0f}'.format(barPublic[i].get_height()),
                xy= (barPublic[i].get_x() + barPublic[i].get_width()/2, barPublic[i].get_height()),
                xytext = (0,3),
                ha = 'center',
                va = 'bottom'
                )

plt.show()





import plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
from scipy import special

py.offline.init_notebook_mode(connected=True)

test = go.Bar(
        x = dataset.index,
        y = dataset['rating'],
        name = 'Rating per Parking')

data = [test]
layout = go.Layout(barmode = 'group')
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'grouped-bar')
