import pandas as pd
import functools
def join_tables(key_value, first_table,secont_table):
    joined = functools.reduce(functools.partial(pd.merge, on = key_value), [first_table,secont_table])
    return joined