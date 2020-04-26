import pandas as pd
import numpy as np
from draft_jenya.restaurants_data_preprocessing import all_restaurant_ids
from draft_jenya.users_data_preprocessing import all_users_ids
from utils.utils_for_files_storing_and_reading import write_df_to_csv


restaurant_ratings = pd.read_csv('../data/rating_final.csv', delimiter =';')

# Printing common statistical characteristics of given ratings
print(restaurant_ratings.iloc[:,2:].describe())

# How many restaurant ids we have across all restaurant data files
all_restaurant_ids_number = len(all_restaurant_ids)

# How many user ids we have across all user data files
all_users_ids_number = len(all_users_ids)

# Building matrix for general, food and service ratings. all matrices are of size m x n, where  m - number of all restaurants ids, n - number of all users ids
# Initially I fill these matrices with default value so that a[i][j]=-1 which means the rating is missing for given i-restaurant and j-user
general_ratings_matrix = np.full((len(all_restaurant_ids), len(all_users_ids)), -1)
food_ratings_matrix = general_ratings_matrix.copy()
service_ratings_matrix = general_ratings_matrix.copy()

# I convert matrices to indexed form where rows are indexed by all restaurant ids and columns are indexed by all user ids
# so it gives possibility to access element of matrix by restaurant id and user id
general_ratings_matrix_df = pd.DataFrame(general_ratings_matrix, columns=all_users_ids, index=all_restaurant_ids)
food_ratings_matrix_df = pd.DataFrame(food_ratings_matrix, columns=all_users_ids, index=all_restaurant_ids)
service_ratings_matrix_df = pd.DataFrame(service_ratings_matrix, columns=all_users_ids, index=all_restaurant_ids)

restaurants_with_ratings_ids = restaurant_ratings['placeID']
users_with_ratings_ids = restaurant_ratings['userID']
restaurants_general_ratings = restaurant_ratings['rating']
restaurants_food_ratings = restaurant_ratings['food_rating']
restaurants_service_ratings = restaurant_ratings['service_rating']
zipped_ratings = zip(
    restaurants_with_ratings_ids,
    users_with_ratings_ids,
    restaurants_general_ratings,
    restaurants_food_ratings,
    restaurants_service_ratings
)

# Filling general, food and service ratings matrices with real values
for restaurant_id, user_id, general_rating, food_rating, service_rating in zipped_ratings:
    general_ratings_matrix_df.loc[restaurant_id, user_id] = general_rating
    food_ratings_matrix_df.loc[restaurant_id, user_id] = food_rating
    service_ratings_matrix_df.loc[restaurant_id, user_id] = service_rating

print(f"Matrix of general ratings:{general_ratings_matrix_df}")
print(f"Matrix of food ratings:{food_ratings_matrix_df}")
print(f"Matrix of service ratings:{service_ratings_matrix_df}")

# write ratings matrices to csv
write_df_to_csv(data_dir="../ratings_matrices", file_name="general_ratings_matrix.csv", data_frame=general_ratings_matrix_df)
write_df_to_csv(data_dir="../ratings_matrices", file_name="food_ratings_matrix.csv", data_frame=food_ratings_matrix_df)
write_df_to_csv(data_dir="../ratings_matrices", file_name="service_ratings_matrix.csv", data_frame=service_ratings_matrix_df)

# I build a matrix to find which restaurant records across all restaurant data files have ratings
# Initially I create the matrix filled with 0 which means that restaurant does not have rating
restaurants_with_ratings_estimation = pd.DataFrame(np.zeros(general_ratings_matrix_df.shape, dtype=int), columns=all_users_ids, index=all_restaurant_ids)

# Then I analyze general ratings matrix which keeps what rating exactly was given to restaurant by user and if this value is missing,
# then -1 must be there, otherwise values 0,1 or 2
# if value is not -1 then we fill restaurants_with_ratings_estimation with 1 value which means that intersection of user and restaurant in this matrix has rating
# To fill this matrix we can use any of general_ratings_matrix_df,food_ratings_matrix_df or service_ratings_matrix_df
# because in rating_final.csv there is no records where one of the general,food or service rating is missing for any row
restaurants_with_ratings_estimation[general_ratings_matrix_df != -1] = 1
print(f"Matrix of ratings between restaurants and users:{restaurants_with_ratings_estimation}")

write_df_to_csv(data_dir="../ratings_matrices", file_name="restaurants_with_ratings_estimation.csv", data_frame=restaurants_with_ratings_estimation)
