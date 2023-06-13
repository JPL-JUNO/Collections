"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-09 16:31:47
"""

import pandas as pd
import numpy as np
from method.frame.util import check_y, x_variable
from pandas import DataFrame, Series


def information_value(df: DataFrame, y: str, x: list | None = None,
                      positive='bad|1',
                      desc: bool | None = None) -> Series:
    """calculate information value for a given dataframe

    Args:
        df (DataFrame): data needed to calculate information value
        y (str): target
        x (list | None, optional): features. Defaults to None.
        positive (str, optional): str used to indicate label. Defaults to 'bad|1'.
        desc (bool | None, optional): order by descending or ascending. Defaults to None.

    Returns:
        Series: Series with x as index, information value as value
    """
    df = df.copy(deep=True)
    if x is not None:
        assert isinstance(x, list), 'x should be a list'
    assert isinstance(y, str), 'y should be a string'
    df = df[[y] + x]
    df = check_y(df, y, positive)
    cols = x_variable(df, y, x)
    # iv_lst = pd.DataFrame({
    #     'variable': cols,
    #     'IV': [iv_calc(df[xi], df[y[0]]) for xi in cols]
    # }).set_index('variable')
    iv = [iv_calc(df[col], df[y]) for col in cols]
    iv_ser = pd.Series(data=iv, index=cols)
    # iv_ser = df[cols].apply(lambda col: iv_calc(df[col], df[y[0]]))

    if desc is not None:
        if desc:
            iv_ser = iv_ser.sort_values(ascending=False)
        else:
            iv_ser = iv_ser.sort_values(ascending=True)
    return iv_ser


def iv_calc(X: Series, y: Series) -> float:
    def neg_pos(df: DataFrame):
        cnt = {'neg': (df['y'] == 0).sum(),
               'pos': (df['y'] == 1).sum()}
        return pd.Series(cnt)

    df_immd = pd.DataFrame({'X': X.astype(str),
                            'y': y}).fillna(value='missing')
    group_by_x_values = df_immd.groupby('X').apply(neg_pos).replace(0, 1)

    # group_res = df_immd.groupby(['X']).value_counts().reset_index(
    # ).pivot_table(index=['X'], columns=['y']).fillna(1)
    # neg_prop = (group_res / group_res.sum())[0][0]
    # pos_prop = (group_res / group_res.sum())[0][1]
    # iv_final = np.sum((neg_prop - pos_prop) / (np.log(neg_prop / pos_prop)))

    pivot_table = group_by_x_values.assign(
        neg_prop=lambda x: x['neg'] / sum(x['neg']),
        pos_prop=lambda x: x['pos'] / sum(x['pos'])).assign(
            iv=lambda x: (x['neg_prop'] - x['pos_prop']) *
        np.log(x['neg_prop'] / x['pos_prop'])
    )
    iv_final = pivot_table['iv'].sum()
    return iv_final
