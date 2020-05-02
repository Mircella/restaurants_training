import pandas as pd
import functools
import numpy as np
from encoding_categorical_variables.relevant_features_analysis import encode_data_frame


def join_tables(key_value, *data_frames):
    joined = functools.reduce(functools.partial(pd.merge, on = key_value), data_frames)
    return joined


def concatenate_tables(*data_frames):
    concatenated_df = pd.concat(data_frames, axis=1, join='inner')
    return concatenated_df


def drop_nan(df):
    df = df.replace('?', np.NaN)
    if {'fax', 'url'}.issubset(df.columns):
        df.drop(['fax','url'], axis = 1, inplace = True)
    df = df.dropna()
    return df


def concat_csv_files(*file_names):
    data_frames = [pd.read_csv(f) for f in file_names]
    concatenated_df = pd.concat(data_frames)
    return concatenated_df


def drop_duplicated_rows_and_columns(data_frame):
    data_frame = data_frame.loc[:, ~data_frame.columns.duplicated()]
    data_frame = data_frame.drop_duplicates()
    return data_frame


def find_duplicated_rows(data_frame):
    duplicated_rows_df = data_frame[data_frame.duplicated()]
    return duplicated_rows_df


def find_unique_records_number_by_column(column, *data_frames):
    data_frames_columns = [df[column] for df in data_frames]
    concatenated_column_values = np.concatenate(data_frames_columns)
    return np.unique(concatenated_column_values)


def inner_join_data_frames_by_column(source_data_frame, restaurants_with_ratings, column_name):
    result = []
    for index, restaurant in source_data_frame.iterrows():
        if index in restaurants_with_ratings[column_name].unique():
            result.append(restaurant)
    return pd.DataFrame(result)


def add_missing_restaurants_with_ratings(source_data_frame, restaurants_with_ratings):
    restaurants_with_ratings_ids = restaurants_with_ratings['placeID'].unique()
    source_data_frame_ids = source_data_frame.index
    is_restaurant_with_rating_in_source_df = [False if elem in source_data_frame_ids else True for elem in restaurants_with_ratings_ids]
    restaurants_with_ratings_ids_not_in_source_df = restaurants_with_ratings_ids[is_restaurant_with_rating_in_source_df]
    result = np.full((len(restaurants_with_ratings_ids_not_in_source_df), len(source_data_frame.columns)), 0)
    return pd.DataFrame(result, index=restaurants_with_ratings_ids_not_in_source_df, columns=source_data_frame.columns)


def merge_and_group(left_df, right_df, merge_column, group_column, estimate_column):
    merged_df = pd.merge(left=left_df, right=right_df, on=merge_column, how="left")
    print(merged_df.shape)
    print(merged_df.head())
    grouped_df = merged_df.groupby(group_column)
    # Calculate the mean ratings
    print(grouped_df[estimate_column].mean())
    print(grouped_df[estimate_column].describe())


def move_index_to_column(data_frame, column_name, reset=False):
    if reset:
        data_frame = data_frame.reset_index()
    data_frame.rename(columns={'index': column_name}, inplace=True)
    return data_frame


def encode_data_frame_and_group_by_index(data_frame):
    data_frame = encode_data_frame(data_frame)
    data_frame = data_frame.groupby(data_frame.index).sum()
    return data_frame