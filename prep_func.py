import pandas as pd
import functools
import numpy as np


def join_tables(key_value, *data_frames):
    joined = functools.reduce(functools.partial(pd.merge, on = key_value), data_frames[0])
    return joined

def drop_nan(df):
    df = df.replace('?', np.NaN)
    df.drop(['fax','url'], axis = 1, inplace = True)
    df = df.dropna()
    return df

