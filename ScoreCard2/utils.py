"""
@Description: Functions that provide convenience methods for
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-27 16:46:14
"""

import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_woe(df: DataFrame, label: str = 'label') -> DataFrame:
    """Calculate Weight of Evidence(WOE) of a given dataframe, which shape is (n, 2)，`label`的内容必须为数值型

    Args:
        df (DataFrame): feature and corresponding label
        label (str, optional): string indicates target. Defaults to 'label'.

    Returns:
        DataFrame: index: different values of feature; columns: different labels and `WOE`
    """
    cols = df.columns.to_list()
    assert label in cols
    df_woe = df.groupby(cols).size().unstack(
        label, fill_value=1).sort_index(axis='columns')
    df_woe = df_woe / df_woe.sum(axis='index')
    # df_woe.div(df_woe.sum(axis='columns'), axis='index')
    df_woe['WOE'] = df_woe.apply(
        lambda x: np.log(x[1] / x[0]), axis='columns')
    return df_woe


def calculate_information_value(data: DataFrame, label: str = 'label') -> float:
    """计算一个分箱的Information Value

    Args:
        data (DataFrame): 需要计算的数据集，应包含两个字段，一个为特征，一个为标签，标签名必须为`label`

    Returns:
        float: 分箱的information value
    """
    df_woe = calculate_woe(data, label)
    df_woe['IV'] = (df_woe[1] - df_woe[0]) * df_woe['WOE']
    return df_woe['IV'].sum()


def filter_by_iv(data, threshold: float = .02, label: str = 'label'):
    cols = data.columns.to_list()
    cols.remove(label)
    col_removed = []
    col_reserved = []
    iv_dict = {}
    for col in cols:
        iv = calculate_information_value(data[[col, label]], label)
        iv_dict[col] = iv
        if iv < threshold:
            col_removed.append(col)
        else:
            col_reserved.append(col)
    return col_reserved, col_removed


def filter_by_diversity(data, diversity: float = .95, label: str = 'label'):
    pass


if __name__ == "__main__":
    df = pd.read_csv('bank.csv', sep=';')
    df['y'] = df['y'].map({'no': 0, 'yes': 1})
    print(filter_by_iv(df, label='y'))
