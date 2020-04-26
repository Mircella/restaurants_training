import pandas as pd

from utils.data_frames_cleaning_functions import join_tables
from utils.data_frames_cleaning_functions import concatenate_tables
from utils.data_frames_cleaning_functions import drop_duplicated_rows_and_columns
from utils.data_frames_cleaning_functions import find_unique_records_number_by_column
from utils.data_frames_cleaning_functions import drop_nan
from utils.utils_for_files_storing_and_reading import write_df_to_csv

# loading users data
user_payment_types = pd.read_csv('../data/userpayment.csv', delimiter =';')
user_cuisine_types = pd.read_csv('../data/usercuisine.csv', delimiter =';')
user_profiles = pd.read_csv('../data/userprofile.csv', delimiter =';')
user_ratings = pd.read_csv('../data/rating_final.csv', delimiter =';')

# Extracting how many payments types users used in restaurants
print(f"Number of unique users with payment type specified:{len(user_payment_types['userID'].unique())}")
print(f"Number of users payment types:{len(user_payment_types['Upayment'].unique())}")

# Extracting how many cuisine types exist in restaurants
print(f"Number of unique users with cuisine type specified:{len(user_cuisine_types['userID'].unique())}")
print(f"Number of preferred cuisine types:{len(user_cuisine_types['Rcuisine'].unique())}")

# Extracting how many there are users who have given profile data
print(f"Number of unique users with given profile data:{len(user_profiles['userID'].unique())}")

# Extracting how many users evaluated restaurants
print(f"Number of users who gave ratings to restaurants:{len(user_ratings['userID'].unique())}")

# How many users do we have across all users data files
all_users_ids = find_unique_records_number_by_column(
    'userID',
    user_payment_types,
    user_cuisine_types,
    user_profiles
)

print(f"All ids of users: {len(all_users_ids)}")

# joining data of users from all tables by their user id to exclude users that do not have any data and will not have impact on the model
joined_user_data = join_tables(
    'userID',
    user_payment_types,
    user_cuisine_types,
    user_profiles,
    user_ratings
)

# see how many records we have after joining
print(f"Number of joined records:{len(joined_user_data)}")

# drop nan from joined user data
joined_user_data = drop_nan(joined_user_data)

# see how many records we have after dropping NaN in joined user data
print(f"Number of joined records after dopping NaN values:{len(joined_user_data)}")

# drop duplicated columns and rows
joined_user_data = drop_duplicated_rows_and_columns(joined_user_data)

# see how many records we have after dropping duplicated columns and rows
print(f"Number of joined records after dropping duplicated columns and rows:{len(joined_user_data)}")

# write joined user data frame to csv file
write_df_to_csv(data_dir="../data", file_name="joined_user_data.csv", data_frame=joined_user_data)

# concatenating data of users from all tables by their user id to exclude users that do not have any data and will not have impact on the model
concatenated_user_data = concatenate_tables(
    user_payment_types,
    user_cuisine_types,
    user_profiles,
    user_ratings
)

# see how many records we have after concatenation
print(f"Number of concatenated records:{len(concatenated_user_data)}")

# drop duplicated columns and rows
concatenated_user_data = drop_duplicated_rows_and_columns(concatenated_user_data)

# see how many records we have after dropping duplicated columns and rows
print(f"Number of concatenated records after dropping duplicated columns and rows:{len(concatenated_user_data)}")

# write concatenated user data frame to csv file
write_df_to_csv(data_dir="../data", file_name="concatenated_user_data.csv", data_frame=concatenated_user_data)

# drop nan from concatenated user data
concatenated_user_data = drop_nan(concatenated_user_data)
# see how many records we have after dropping NaN in concatenated user data
print(f"Number of concatenated records after dopping NaN values:{len(concatenated_user_data)}")

# write concatenated user data frame to csv file
write_df_to_csv(data_dir="../data", file_name="concatenated_user_data.csv", data_frame=joined_user_data)


