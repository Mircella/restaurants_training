#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In this notebook I am going to play with input data, compare general statistics about it and see what can be potentially used 
#to find more relevant features which impact rating of restaurants.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils_for_files_storing_and_reading import write_df_to_csv
from utils.functions_for_encoding import drop_columns


# In[2]:


# Initially, the downloaded data has different types of delimiters. All data sets were saved in the correct format in the folder
# clean_data. If the data is already in that folder, some steps can be skipped. In this case, start to run code in 6 and 8 boxes
# to get the data of restaurants and users.


# In[3]:


# loading restaurants data
restaurant_payment_types = pd.read_csv('raw_data/chefmozaccepts.csv', delimiter =';')
restaurant_cuisine_types = pd.read_csv('raw_data/chefmozcuisine.csv', delimiter =';')
restaurant_working_hours = pd.read_csv('raw_data/chefmozhours.csv', delimiter =',')
restaurant_parking = pd.read_csv('raw_data/chefmozparking.csv', delimiter =';')
restaurant_geo_places = pd.read_csv('raw_data/geoplaces.csv', delimiter =';', encoding='latin-1')
ratings = pd.read_csv('raw_data/rating_final.csv', delimiter =';')


# In[4]:


# Loading users data
# The user data will be used later.
user_payment_types = pd.read_csv('raw_data/userpayment.csv', delimiter =';')
user_cuisine_types = pd.read_csv('raw_data/usercuisine.csv', delimiter =';')
user_profiles = pd.read_csv('raw_data/userprofile.csv', delimiter =';')


# In[5]:


# I created a new directory with clean .csv files that have unified formatting as input data have different delimiters.
write_df_to_csv('clean_data', 'restaurant_payment_types.csv', restaurant_payment_types)
write_df_to_csv('clean_data', 'restaurant_cuisine_types.csv', restaurant_cuisine_types)

# As restaurant_working_hours data set has mixed types of delimiters, I rewrite its columns 'hours' and 'days' to be consistent
# in formatting with the rest of data sets.

for i, series in restaurant_working_hours.iterrows():
    hours = restaurant_working_hours.loc[i, 'hours'][0:len(restaurant_working_hours.loc[i, 'hours']) - 1]
    restaurant_working_hours.loc[i, 'hours'] = hours
    hours = restaurant_working_hours.loc[i, 'days'][0:len(restaurant_working_hours.loc[i, 'days']) - 1]
    restaurant_working_hours.loc[i, 'days'] = hours

write_df_to_csv('clean_data', 'restaurant_working_hours.csv', restaurant_working_hours)
write_df_to_csv('clean_data', 'restaurant_parking_types.csv', restaurant_parking)
write_df_to_csv('clean_data', 'restaurant_geo_places.csv', restaurant_geo_places)
write_df_to_csv('clean_data', 'restaurant_ratings.csv', ratings)


# In[6]:


# All restaurant related data in the correct format will be used for the further analysis.
r_payment = pd.read_csv('clean_data/restaurant_payment_types.csv')
r_cuisine = pd.read_csv('clean_data/restaurant_cuisine_types.csv')
r_hours = pd.read_csv('clean_data/restaurant_working_hours.csv')
r_parking = pd.read_csv('clean_data/restaurant_parking_types.csv')
r_general = pd.read_csv('clean_data/restaurant_geo_places.csv')
ratings = pd.read_csv('clean_data/restaurant_ratings.csv')


# In[7]:


# Here I also save clean user data frames with default delimiter=','. 
write_df_to_csv('clean_data', 'user_payment_types.csv', user_payment_types)
write_df_to_csv('clean_data', 'user_cuisine_types.csv', user_cuisine_types)
write_df_to_csv('clean_data', 'user_profiles.csv', user_profiles)


# In[8]:


# All user related data in the correct format will be used for the further analysis.
u_payment = pd.read_csv('clean_data/user_payment_types.csv')
u_cuisine = pd.read_csv('clean_data/user_cuisine_types.csv')
u_profiles = pd.read_csv('clean_data/user_profiles.csv')


# In[9]:


# Some rows in the data have the same value but written in a different format, e.g. 'VISA' vs 'Visa'. 
r_payment['Rpayment'].unique()


# In[10]:


# I use the below function to correct the data.
def lowercase(df, *columns):
    for column in columns:
        df[column] = df[column].str.lower()


# In[11]:


# The cuisine data
# There are no duplicates or some values which can be merged so I leave it like it is.
lowercase(r_cuisine, 'Rcuisine')
sorted(r_cuisine['Rcuisine'].unique())


# In[12]:


# Extracting how many cuisine types exist in restaurants
print(f"Number of unique restaurants with cuisine type specified:{len(r_cuisine['placeID'].unique())}")
print(f"Number of cuisine types:{len(r_cuisine['Rcuisine'].unique())}")


# In[13]:


# The barplot for cuisines
# As there are 59 different types of cuisine, I display top 10 by frequency. The survey was conducted in Mexico, so no
# surprise the most frequent cuisine is Mexican.

cuisine_count = r_cuisine['Rcuisine'].value_counts()
plt.figure(figsize=(16, 6))
sns.set(style="darkgrid")
sns.barplot(cuisine_count.index[:10], cuisine_count.values[:10], alpha=0.9)
plt.title('Top 10 Frequent Cuisines', fontsize = 16)
plt.ylabel('Number of Occurrences', fontsize=14)
plt.xlabel('Cuisine', fontsize=14)
plt.show()


# In[14]:


# The parking data
# There are no duplicates or some values which can be merged so I leave it like it is.
lowercase(r_parking, 'parking_lot')
sorted(r_parking['parking_lot'].unique())


# In[15]:


# The barplot for parking
# Almost 350 restaurant do not have a parking option. I'll check whether the parking plays a significant role for a rating. If
# so, a restaurant might include it in their services to increase its rating.

parking_count = r_parking['parking_lot'].value_counts()
plt.figure(figsize=(16, 6))
sns.set(style="darkgrid")
sns.barplot(parking_count.index, parking_count.values, alpha=0.9)
plt.title('Parking Options', fontsize = 16)
plt.ylabel('Number of Occurrences', fontsize=14)
plt.xlabel('Parking', fontsize=14)
plt.show()


# In[16]:


# The payment data
# 'Visa' was changed to 'visa. There are no more duplicates or some values to be changed.
lowercase(r_payment, 'Rpayment')
sorted(r_payment['Rpayment'].unique())


# In[17]:


# As there are many options to pay in restaurants, firstly, I check what options are the most frequent. 
r_payment['Rpayment'].value_counts()


# In[18]:


# The bar plot for payment
payment_count = r_payment['Rpayment'].value_counts()
plt.figure(figsize=(16, 6))
sns.set(style="darkgrid")
sns.barplot(payment_count.index[:6], payment_count.values[:6], alpha=0.9)
plt.title('Payment Options', fontsize=16)
plt.ylabel('Number of Occurrences', fontsize=14)
plt.xlabel('Payment', fontsize=14)
plt.show()


# In[19]:


# The working schedule data
lowercase(r_hours, 'days')
r_hours['hours'].value_counts()


# In[20]:


# There are 273 different shifts over three types of working days. Many working hours are intersected with each other. As there 
# are many different options of working hours of restaurants, it's been decided not to consider this variable. If a client 
# evaluated a restaurant, it assumes he attended it and the working shift is not an issue for the client. Moreover,the data does
# not state what hours are more profitable for a restaurant and it's not clear how it can be evaluated. 

pd.pivot_table(r_hours, values = ['hours'], index = 'hours', columns = 'days',aggfunc = 'count')


# In[21]:


# The general data
# Some columns in the r_general will be removed. Those columns, in my opinion, will not affect the rating and/or the
# visualization of those columns is irrelevant and does not shed a light on the data.

r_general = drop_columns(r_general, 'latitude', 'longitude', 'the_geom_meter', 'address',
             'fax','zip','url', 'name')
lowercase(r_general, 'city', 'state', 'country', 'alcohol', 'smoking_area', 
          'dress_code', 'accessibility', 'price', 'Rambience', 'franchise', 'area', 
          'other_services')


# In[22]:


# Checking the remaining columns
# I will start from country.

r_general['country'].value_counts()


# In[23]:


# Mexico is a country for the majority of restaurants. Some restaurants with a missing country still have state and/or city 
# indicatiting it's still Mexico. A few records have neither country nor state nor city. Nevertheless, doesn't matter as the
# survey was conducted for Mexican restaurants. As this value is the same for all places, the column 'country' will be dropped 
# from the dataset.
r_general = drop_columns(r_general,'country')


# In[24]:


# When checking state, some values caught my attention. It's slp, san luis potosi, s.l.p., san luis potos. They all refer to one
# state San Luis Potose. I decided to replace all those values by 'san luis potosi'. '?' were replaced by 'Nan'

print(r_general['state'].value_counts())

r_general['state'] = r_general['state'].replace(['slp', 'san luis potosi', 's.l.p.', 'san luis potos'],'san luis potosi')
r_general['state'] = r_general['state'].replace('?', 'Nan')

print(f"\n{r_general['state'].value_counts()}")


# In[25]:


# Some city values can be merged like it was done for the 'state'. '?' were replaced by 'Nan'. You can see how the column 'city'
# changed after the replacement.

print(r_general['city'].value_counts())

r_general['city'] = r_general['city'].replace(['san luis potosi', 'slp', 's.l.p.', 's.l.p', 'san luis potos','san luis potosi '],'san luis potosi')
r_general['city'] = r_general['city'].replace(['victoria', 'victoria ', 'ciudad victoria', 'cd victoria', 'cd. victoria'],
                                              'ciudad victoria')
r_general['city'] = r_general['city'].replace('?', 'Nan')

print(f"\n{r_general['city'].value_counts()}")


# In[26]:


# The column 'alcohol' won't be changed.  
r_general['alcohol'].value_counts()


# In[27]:


# The column 'smoking_area'
# I checked what options are in the smoking_area column. I deciced to have two categories:
# 1) not permitted (none, not permitted) and 2) permitted (only at bar, section, permitted)
print(r_general['smoking_area'].value_counts())
r_general['smoking_area'] = r_general['smoking_area'].replace(['none'],'not permitted')
r_general['smoking_area'] = r_general['smoking_area'].replace(['only at bar', 'section'],'permitted')
print(f"\n{r_general['smoking_area'].value_counts()}")
      
# I assume smoking_area and alcohol will play a significant role for a rating. This can be a decisive factor when choosing and 
# evaluating a restaurant, especially for smokers and/or people who refuse to go to a restaurant if they don’t serve alcohol.
# It's good there are different options for those two variables. The majority of restaurants do not serve alcohol and do not
# permit smoking. It is interesting to check whether the lack of the options decreases the rating of a restaurant or 
# clients do not consider them as important factors.


# In[28]:


# The column 'dress_code'
# I decided to unite informal and casual and eventually have only two options for dress_code: informal and formal.

print(r_general['dress_code'].value_counts())
r_general['dress_code'] = r_general['dress_code'].replace(['casual'],'informal')
print(f"\n{r_general['dress_code'].value_counts()}")
      
# There are only 2 restaurants with formal clothes. I assume it will not affect the rating of a restaurant if almost all of them
# have the same value per one category.


# In[29]:


# The values of rest columns from r_general. After checking the values, I decided not to change them.

print(f"\nThe options of accessibility are: \n{r_general['accessibility'].value_counts()}")
print(f"\nThe options of price are: \n{r_general['price'].value_counts()}")
print(f"\nThe options of Rambience are: \n{r_general['Rambience'].value_counts()}")
print(f"\nThe options of franchise are: \n{r_general['franchise'].value_counts()}")
print(f"\nThe options of area are: \n{r_general['area'].value_counts()}")
print(f"\nThe options of other_services are: \n{r_general['other_services'].value_counts()}")


# In[30]:


# Barplots for r_general. 
# As there are many categories in this dataset, I decided to write the general function to get the plot for "X' column from
# r_general.
def plot_general(column):
    count = r_general[column].value_counts()
    plt.figure(figsize=(16, 6))
    sns.set(style="darkgrid")
    sns.barplot(count.index, count.values, alpha=0.9)
    plt.title(f'{column.title()} Options', fontsize = 16)
    plt.ylabel('Number of Occurrences', fontsize=14)
    plt.xlabel(f'{column.title()}', fontsize=14)
    plt.show()  


# In[31]:


plot_general('alcohol')
plot_general('smoking_area')
plot_general('price')

# If needed, it's possible to make a bar plot for any variable. I checked three of them.


# In[32]:


# Rating dataset
# I decided to check the correlation between different rating from ratings dataset. I created a new dataframe to see the average
# rating of each rating grouped by placeID. I got a table of 130 restaurants with their ratings.

rating_mean = ratings.groupby('placeID')['rating', 'food_rating', 'service_rating'].mean().reset_index()
print(rating_mean)

print(rating_mean.iloc[:,1:].describe())

correlation = rating_mean.iloc[:,1:].corr()
plt.figure(figsize=(16, 6))
sns.heatmap(correlation)

print(f'\nThe rating correlation: \n{correlation}')

# I assume rating and service_rating will be affected by the same variables.


# In[33]:


# I saved the clean data sets + rating dataset (to have all sets in one folder) to csv files which will be used later.
write_df_to_csv('from_data_visualization', 'restaurant_cuisine.csv', r_cuisine)
write_df_to_csv('from_data_visualization', 'restaurant_general.csv', r_general)
write_df_to_csv('from_data_visualization', 'restaurant_hours.csv', r_hours)
write_df_to_csv('from_data_visualization', 'restaurant_parking.csv', r_parking)
write_df_to_csv('from_data_visualization', 'restaurant_payment.csv', r_payment)
write_df_to_csv('from_data_visualization', 'ratings.csv', ratings)
write_df_to_csv('from_data_visualization', 'rating_mean.csv', rating_mean)


# In[34]:


# Here I want to merge data of restaurants from different data sets by placeID with ratings and see how different features 
# affect a rating. I will continue to work with the clean datasets. 
# As I saved clean data sets, I can use those sets any time I need it.

# I will merge the rating dataset with other frames and then display the data on bar charts. I want to see how rating is
# different depending on column values. For this, I am creating the list of rating which will be used for bar charts +
# additional fixed arguments used for all plots.

subjects = ['rating', 'food_rating', 'service_rating']
indx = np.arange(len(subjects))
bar_width = 0.25

# As the scale of rating is narrow [0,2], I want to see the numbers of a rating on the plot. To do this, the following
# function will be used.

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# In[35]:


# Parking and Rating data sets
parking_rating = pd.merge(rating_mean,r_parking,on = 'placeID', how = 'left')
df_bar_plot = parking_rating.groupby('parking_lot')[subjects].mean().T


# In[36]:


# Parking and Rating plot
fig, ax = plt.subplots(figsize = (16,7))
rects1 = plt.bar(indx + 0.00, df_bar_plot.iloc[:,0],bar_width, label = 'none')
rects2 = plt.bar(indx + 0.22, df_bar_plot.iloc[:,1],bar_width, label = 'public')
rects3 = plt.bar(indx + 0.44, df_bar_plot.iloc[:,2],bar_width, label = 'valet parking')
rects4 = plt.bar(indx + 0.66, df_bar_plot.iloc[:,3],bar_width, label = 'yes')

ax.set_ylabel('Scores', fontsize = 14)
ax.set_title('Rating for Parking Options', fontsize = 16)
ax.set_xticks(indx)
ax.set_xticklabels(subjects, fontsize = 14)
ax.legend(fontsize = 15, loc = (1.05, 0.4))

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)


plt.show()


# In[37]:


# I already know there are 130 restaurants which were evaluated by users. I want to know the number of users who gave the rating
# and who have the user details.

def find_unique_records_number_by_column(column, *data_frames):
    data_frames_columns = [df[column] for df in data_frames]
    concatenated_column_values = np.concatenate(data_frames_columns)
    return np.unique(concatenated_column_values)

all_users_ids = find_unique_records_number_by_column(
    'userID',
    user_payment_types,
    user_cuisine_types,
    user_profiles
)

print(f"The number of users: {len(all_users_ids)}")


# In[38]:


# There are 138 users I have the data about. Not all of them might evaluate restaurants. Firstly, I will clean the uder data like
# it was done for the restaurant data. 

# The user_profile data

u_profiles.head()


# In[39]:


# Drop redundant columns from u_profiles
u_profiles = drop_columns(u_profiles, 'latitude', 'longitude', 'birth_year', 'color', 'weight', 'height')


# In[40]:


u_profiles.head()


# In[41]:


lowercase(u_profiles, 'smoker', 'drink_level', 'dress_preference', 'ambience', 'transport', 'marital_status', 'hijos', 
          'interest', 'personality', 'religion', 'activity', 'budget')


# In[42]:


# I am going to check values in each column and check if some values can be combined or need to be replaced. 

print(f"\n{u_profiles['smoker'].value_counts()}")
print(f"\n{u_profiles['drink_level'].value_counts()}")
print(f"\n{u_profiles['dress_preference'].value_counts()}")
print(f"\n{u_profiles['ambience'].value_counts()}")
print(f"\n{u_profiles['transport'].value_counts()}")
print(f"\n{u_profiles['marital_status'].value_counts()}")
print(f"\n{u_profiles['hijos'].value_counts()}") 
print(f"\n{u_profiles['interest'].value_counts()}")
print(f"\n{u_profiles['personality'].value_counts()}")
print(f"\n{u_profiles['religion'].value_counts()}")
print(f"\n{u_profiles['activity'].value_counts()}")
print(f"\n{u_profiles['budget'].value_counts()}")


# In[43]:


# After analyzyng the above result, I decided to replace some values to reduce the number of options.

u_profiles['dress_preference'] = u_profiles['dress_preference'].replace(['formal','elegant'], 'formal')
u_profiles['transport'] = u_profiles['transport'].replace(['public','on foot'], 'not car owner')
u_profiles['marital_status'] = u_profiles['marital_status'].replace(['single','widow'], 'not married')
u_profiles['hijos'] = u_profiles['hijos'].replace(['kids','dependent'], 'with kids') #hijos means children
u_profiles['hijos'] = u_profiles['hijos'].replace(['independent'], 'without kids')


# In[44]:


# Payment User data

lowercase(u_payment, 'Upayment')
print(f"\n{u_payment['Upayment'].value_counts()}")


# In[45]:


# Cuisine User data

lowercase(u_cuisine, 'Rcuisine')
sorted(u_cuisine['Rcuisine'].unique())


# In[46]:


# Some values are ambiguous and it's unclear how they can affect the rating. However, there might be a strong correlation between
# some restaurants and users features, e.g. smoking_area of a restaurant and smoker of a user. I will check how those features
# are dependent on each other. 


# In[47]:


# Merging rating and smoking details of restaurants and users

smoking_df = pd.merge(pd.merge(ratings,  r_general[['placeID', 'smoking_area']], on = 'placeID', how = 'left'),
                      u_profiles[['userID', 'smoker']], on = 'userID', how = 'left')


# In[48]:


# Mean rating grouped by users, smoker, and smoking_area

smoking_rating_mean = smoking_df.groupby(['userID', 'smoker', 'smoking_area'])['rating', 'service_rating'].mean().reset_index()


# In[49]:


# How many smokers and non-smokers among users

counts = smoking_rating_mean.groupby('smoker')['userID'].nunique()
labels = ['N/A', 'Non-smoker', 'Smoker']
fig1, ax1 = plt.subplots(figsize = (16,7))
plt.title("Smoker Distribution among Users", fontsize = 16)
explode = (0,0.1,0)
ax1.pie(counts,explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.show()


# In[50]:


smoking_stats = pd.pivot_table(smoking_rating_mean, values = ['rating', 'service_rating'], index = 'smoking_area', 
               columns = 'smoker',aggfunc = 'mean')

print(smoking_stats )

# I am doing this as there were only one variable, i.e. smoking area, affecting the rating. It's for simplicity only and to
# check how smoking_area affects users' rating. For this variable, I consider rating and service rating although they are highly
# correlated. 
# As per below table, I assume smoking_area might have an impact on the rating but along with other variables.
# Only after the model implementation I can state whether there is an evidence this variable affects the result or not.


# In[51]:


# Merging rating and drinking details of restaurants and users

drinking_df  = pd.merge(pd.merge(ratings,  r_general[['placeID', 'alcohol']], on = 'placeID', how = 'left'),
                      u_profiles[['userID', 'drink_level']], on = 'userID', how = 'left')

# Mean rating grouped by users, alcohol, and drink_level

drinking_rating_mean = drinking_df.groupby(['userID', 'alcohol', 'drink_level'])['rating', 'food_rating',
                                                                                 'service_rating'].mean().reset_index()


# In[52]:


# Drink_level stats

counts = drinking_rating_mean.groupby('drink_level')['userID'].nunique()
labels = ['abstemious', 'casual drinker', 'social drinker']
fig1, ax1 = plt.subplots(figsize = (16,7))
plt.title("Drink Level among Users", fontsize = 16)
explode = (0,0,0)
ax1.pie(counts,explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.show()


# In[53]:


drinking_stats = pd.pivot_table(drinking_rating_mean, values = ['rating', 'food_rating','service_rating'], index = 'alcohol', 
               columns = 'drink_level',aggfunc = 'mean')

print(drinking_stats)


# In[54]:


# How budget and price affect the rating
# Merging rating and budget/price details of restaurants and users

budget_df  = pd.merge(pd.merge(ratings,  r_general[['placeID', 'price']], on = 'placeID', how = 'left'),
                      u_profiles[['userID', 'budget']], on = 'userID', how = 'left')
budget_rating_mean = budget_df.groupby(['userID', 'price', 'budget'])['rating', 'food_rating',
                                          'service_rating'].mean().reset_index()


# In[55]:


# Budget stats

counts = budget_rating_mean.groupby('budget')['userID'].nunique()
labels = ['N/A', 'high', 'low', 'medium']
fig1, ax1 = plt.subplots(figsize = (16,7))
plt.title("Budget Level among Users", fontsize = 16)
explode = (0,0,0,0.1)
ax1.pie(counts,explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.show()


# In[56]:


budget_stats = pd.pivot_table(budget_rating_mean, values = ['rating', 'food_rating','service_rating'], index = 'price', 
               columns = 'budget',aggfunc = 'mean')

print(budget_stats)

# As expected the lowest rating is for low price restaurants, especially for service_raiting among users of all budget types.


# In[57]:


# There are too many variations that can be checked, including > 2 variables from both restaurants and users data. It can be done
# through more profound analysis basing not only on numbers but also on assumptions knowing business area.

# I will not check all of them. The above graphs can be used for better understanding of your clients profiles. If the majority
# of clients are smokers, a restaurant's owner might dig deeper into the smoking_area options to check what they prefer more: 
# separate section, at bar, or something else. 

# A restaurant's owner might consider to differentiate the restaurant from others by changing a few restaurants features so 
# clients will prefer his restaurant over others. For example, among 130 restaurants, only 2 restaurants requires formal dress 
# code, whereas the rest is for informal - I suppose formal clothes is accepted as well. According to the users' profiles 32.6%
# people prefer elegant or formal clothes, which I merge into one group 'not casual'. If there were more restaurants with formal
# dress code, it could be analyzed whether such restaurants receive higher rating as it is considered a higher level in comparison
# to others. To make such a business decision, it's really needed to know the industry and how it works, but the data can help
# with this. 

user_clothes_df = u_profiles[['userID', 'dress_preference']]
clothes_counts = user_clothes_df.groupby('dress_preference')['userID'].nunique()
labels = ['N/A', 'informal', 'no preference', 'not causal']
fig1, ax1 = plt.subplots(figsize = (16,7))
plt.title("Clothes Preferences among Users", fontsize = 16)
explode = (0,0,0,0.1)
ax1.pie(clothes_counts,explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.show()


# In[58]:


# The above example is more when business and statistics should work together to make a decision. As I am not aware of nuances of
# this industry, I cannot state whether it will be a good decision or not. I may assume it's not only one variable which should
# be considered.

# Lastly, I checked the average rating per four different categories (franchise, area, rambience, other_services) from 
# r_general as those categories were selected as features which affect most the rating of restaurants. Its selection
# will be explained in details in the section 'methodology'.

subjects = ['rating', 'food_rating', 'service_rating']
indx = np.arange(len(subjects))
bar_width = 0.35


# In[59]:


# Franchise Rating
# Franchise has a higher score than non-franchise per each rating type.

franchise_rating = pd.merge(ratings,  r_general[['placeID', 'franchise']], on = 'placeID', how = 'left')
franchise_rating_mean = franchise_rating.groupby(['franchise'])['rating', 'food_rating', 'service_rating'].mean().T

fig, ax = plt.subplots(figsize = (16,7))
rects1 = plt.bar(indx + 0.00, franchise_rating_mean.iloc[:,0],bar_width, label = 'not franchise')
rects2 = plt.bar(indx + 0.35, franchise_rating_mean.iloc[:,1],bar_width, label = 'franchise')
ax.set_ylabel('Scores', fontsize = 14)
ax.set_title('Rating for Franchise Options', fontsize = 16)
ax.set_xticks(indx)
ax.set_xticklabels(subjects, fontsize = 14)
ax.legend(fontsize = 15, loc = (1.05, 0.4))

autolabel(rects1)
autolabel(rects2)

plt.show()


# In[60]:


# Area Rating
# There is a significant difference between close and open area for general and service ratings.

area_rating = pd.merge(ratings,  r_general[['placeID', 'area']], on = 'placeID', how = 'left')
area_rating_mean = area_rating.groupby(['area'])['rating', 'food_rating', 'service_rating'].mean().T

fig, ax = plt.subplots(figsize = (16,7))
rects1 = plt.bar(indx + 0.00, area_rating_mean.iloc[:,0],bar_width, label = 'closed')
rects2 = plt.bar(indx + 0.35, area_rating_mean.iloc[:,1],bar_width, label = 'open')
ax.set_ylabel('Scores', fontsize = 14)
ax.set_title('Rating for Area Types Options', fontsize = 16)
ax.set_xticks(indx)
ax.set_xticklabels(subjects, fontsize = 14)
ax.legend(fontsize = 15, loc = (1.05, 0.4))

autolabel(rects1)
autolabel(rects2)

plt.show()


# In[61]:


# Rambience Rating

rambience_rating = pd.merge(ratings,  r_general[['placeID', 'Rambience']], on = 'placeID', how = 'left')
rambience_rating_mean = rambience_rating.groupby(['Rambience'])['rating', 'food_rating', 'service_rating'].mean().T

fig, ax = plt.subplots(figsize = (16,7))
rects1 = plt.bar(indx + 0.00, rambience_rating_mean.iloc[:,0],bar_width, label = 'familiar')
rects2 = plt.bar(indx + 0.35, rambience_rating_mean.iloc[:,1],bar_width, label = 'quiet')
ax.set_ylabel('Scores', fontsize = 14)
ax.set_title('Rating for Rambience Types Options', fontsize = 16)
ax.set_xticks(indx)
ax.set_xticklabels(subjects, fontsize = 14)
ax.legend(fontsize = 15, loc = (1.05, 0.4))

autolabel(rects1)
autolabel(rects2)

plt.show()


# In[62]:


# Other_services Rating
# Restaurants without any additional services provided have the lowest rating in all ratings and the difference is really 
# significant.

other_services_rating = pd.merge(ratings,  r_general[['placeID', 'other_services']], on = 'placeID', how = 'left')
other_services_rating_mean = other_services_rating.groupby(['other_services'])['rating', 'food_rating', 
                                                                               'service_rating'].mean().T

fig, ax = plt.subplots(figsize = (16,7))
rects1 = plt.bar(indx + 0.00, other_services_rating_mean.iloc[:,0],bar_width, label = 'internet')
rects2 = plt.bar(indx + 0.25, other_services_rating_mean.iloc[:,1],bar_width, label = 'none')
rects3 = plt.bar(indx + 0.5, other_services_rating_mean.iloc[:,2],bar_width, label = 'variety')
ax.set_ylabel('Scores', fontsize = 14)
ax.set_title('Rating for Other Services Types Options', fontsize = 16)
ax.set_xticks(indx)
ax.set_xticklabels(subjects, fontsize = 14)
ax.legend(fontsize = 15, loc = (1.05, 0.4))

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()


# In[63]:


# This section helps to understand the data easier rather than checking all rows in different dataset. It's easier to combine
# two or more dataset on the required columns and check for any correlation.

# The next section 'Modelling' is to build models to predict how restaurants will be evaluated by people basing on restaurants 
# features.


# In[ ]:




