import pandas as pd
import functools
import numpy as np


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


def extract_restaurants_with_ratings(source_data_frame, restaurants_with_ratings):
    result = []
    for index, restaurant in source_data_frame.iterrows():
        if restaurant['placeID'] in restaurants_with_ratings:
            result.append(restaurant)
    return result