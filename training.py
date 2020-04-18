import pandas as pd


def extract_restaurants_with_ratings(source_data_frame, restaurants_with_ratings):
    result = []
    for index, restaurant in source_data_frame.iterrows():
        if restaurant['placeID'] in restaurants_with_ratings:
            result.append(restaurant)
    return result

chefmozaccepts = pd.read_csv('data/chefmozaccepts.csv', delimiter=';')
chefmozcuisine = pd.read_csv('data/chefmozcuisine.csv', delimiter=';')
chefmozhours = pd.read_csv('data/chefmozhours.csv', delimiter=',')
chefmozparking = pd.read_csv('data/chefmozparking.csv', delimiter=';')
geoplaces = pd.read_csv('data/geoplaces.csv', delimiter=';', encoding='latin-1')

rating_final = pd.read_csv('data/rating_final.csv', delimiter=';')

usercuisine = pd.read_csv('data/usercuisine.csv', delimiter=';')
userpayment = pd.read_csv('data/userpayment.csv', delimiter=';')
userprofile = pd.read_csv('data/userprofile.csv', delimiter=';')

unique_payment_types = chefmozaccepts['Rpayment'].unique()
unique_cuisine_types = chefmozcuisine['Rcuisine'].unique()
unique_parking_types = chefmozparking['parking_lot'].unique()

restaurants_number_1 = len(chefmozaccepts['placeID'].unique())
restaurants_number_2 = len(chefmozcuisine['placeID'].unique())
restaurants_number_3 = len(chefmozhours['placeID'].unique())
restaurants_number_4 = len(chefmozparking['placeID'].unique())

restaurants_with_ratings = rating_final['placeID'].unique()
restaurants_with_ratings_number = len(restaurants_with_ratings)

restaurants_with_payments_and_ratings = extract_restaurants_with_ratings(chefmozaccepts, restaurants_with_ratings)
restaurants_with_cuisine_and_ratings = extract_restaurants_with_ratings(chefmozaccepts, restaurants_with_ratings)

print(len(restaurants_with_payments_and_ratings))

users_number_1 = len(usercuisine['userID'].unique())
users_number_2 = len(userpayment['userID'].unique())
users_number_3 = len(userprofile['userID'].unique())
users_number_4 = len(rating_final['userID'].unique())
print(users_number_1)
print(users_number_2)
print(users_number_3)
print(users_number_4)
