import pandas as pd
import numpy as np
df = pd.DataFrame({'feature1': ['A', 'B', 'A', 'B', 'C', 'C', 'D', 'A'],
                   'label': [1, 1, 0, 0, 1, 0, 1, 0]})

df_immd = df.groupby(df.columns.tolist()).size().unstack(
    'label', fill_value=1).sort_index(axis='columns')
# .rename_axis(index=None, columns=None)
df_immd = df_immd / df_immd.sum(axis='index')
df_immd['WOE'] = df_immd.apply(lambda x: np.log(x[1] / x[0]), axis='columns')
print(df_immd)

df = pd.read_csv('bank.csv', sep=';')
df['y'] = df['y'].map({'no': 0, 'yes': 1})
from utils import calculate_information_value
calculate_information_value(df[['education', 'y']], 'y')
