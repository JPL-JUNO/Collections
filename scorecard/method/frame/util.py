"""
@Description:
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-08 10:17:21
"""
import pandas as pd
import numpy as np
import re
import warnings
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from typing import Dict


def feature_zip(data: DataFrame, var: str, c: float = .3, if0: bool = False,
                inplace: bool = False, e: int | None = None, plot: bool = False,
                label: str = 'label', duplicate_check: bool = False):
    unique_0 = len(set(data[var]))
    if plot:
        data_0 = data[[label, var]]
    isn0_mask = (data[var] != 0).astype(int)
    if if0:
        if0_tem_mask = (data[var] == 0).astype(int)
        if duplicate_check:
            assert var + 'if0' not in data.columns, 'duplicated variable name...'
        data[var + 'if0'] = if0_tem_mask

    if e is None:
        p90 = data.loc[data[var] != 0, var].quantile(c)
        p90 = 1 if math.isnan(p90) else abs(p90)
        e = math.floor(math.log(p90, 10))
        sig = [-1 if i < 0 else 1 for i in data[var]]
        tem = [round(abs(i), -e) if abs(i) >= 10**e /
               2 else 10**e / 2 for i in data[var]]
        tem = list(map(lambda x, y, z: x * y * z, tem, sig, isn0_mask))
        unique_1 = len(set(tem))
    else:
        assert isinstance(e, int), 'e must be an integer or None'
        sig = [-1 if i < 0 else 1 for i in data[var]]
        tem = [round(abs(i), -e) if abs(i) >= 10**e /
               2 else 10**e / 2 for i in data[var]]
        tem = list(map(lambda x, y, z: x * y * z, tem, sig, isn0_mask))
    if inplace:
        data[var] = tem
        var_n = var
    else:
        if duplicate_check:
            assert var + '_zip_e' + \
                str(e) not in data.columns, 'duplicated variable name...'
        data[var + '_zip_e' + str(e)] = tem
        var_n = var + '_zip_e' + str(e)
    if plot:
        plt.figure(12, figsize=(12, 6))
        plt.subplot(221)
        sns.displot(data_0[var][data_0[label] == 1].drop(),
                    kde=False, color='red')
        sns.displot(data_0[var][data_0[label] == 0].drop(),
                    kde=False, color='blue')
        plt.subplot(221)
        sns.displot(data[var_n][data[label] == 1].drop(),
                    kde=False, color='red')
        sns.displot(data[var_n][data[label] == 0].drop(),
                    kde=False, color='blue')
    print('{0} has been zipped from {1:>5} to {2:>5} with replace is {3}...'.format(
        var, unique_0, unique_1, str(inplace)))


def str_to_list(x: str | None):
    if x is not None and isinstance(x, str):
        x = [x]
    return x


def check_y(data, y: str, positive) -> DataFrame:
    positive = str(positive)
    assert isinstance(data, DataFrame), '数据类型需要为pd.DataFrame'
    assert data.shape[1] > 2, '数据的变量数必须大于2'
    assert y in data.columns, 'target不在数据中'

    if data[y].isnull().sum():
        warnings.warn('target包含NULL, 移除该样本')
        data = data.dropna(subset=y)
    # 将 y 转为整数
    if is_numeric_dtype(data[y]):
        data[y] = data[y].apply(lambda x: x if pd.isnull(x) else int(x))
    # unique_y = np.unique(data[y].values)
    unique_y = set(data[y].values)
    assert len(unique_y) == 2, '预测变量不符合二分类'

    if any([re.search(positive, str(v)) for v in unique_y]):
        y1 = data[y]
        y2 = data[y].apply(lambda x: 1 if str(
            x) in re.split('\|', positive) else 0)
        if (y1 != y2).any():
            data[y] = y2
            # data[data.columns[y]] = y2
            warnings.warn(
                '默认修改 positive value \{}\ 为 1, negative value 为 0 '.format(y))
    else:
        raise ValueError('positive value 未被正确声明')
    return data


def x_variable(data: DataFrame, y: str,
               x: list | tuple,
               var_skip: str = None) -> list | tuple:
    """determine feature columns remained

    Args:
        data (DataFrame): data
        y (str): target
        x (list | tuple): columns specified, must be in list or tuple
        var_skip (str, optional): some column remove manually. Defaults to None.

    Returns:
        list | tuple: columns
    """
    y = str_to_list(y)
    if var_skip is not None:
        y = y + str_to_list(var_skip)
    col_all = list(set(data.columns).difference(set(y)))
    if x is None:
        # 如果没有指定变量，则数据中的字段名即为变量
        x = col_all
    else:
        # 如果手动指定了便令，则判断指定的变量是否与数据中字段存在交集，存在则为交集，
        # 没有交集则为数据中的字段
        assert isinstance(
            x, (list, tuple)), '手动指定的字段格式不正确，必须是列表(list)或者元组(tuple)'
        x_inter = set(x).intersection(set(col_all))
        x_except = set(x).difference(set(col_all))
        if len(x_except) > 0:
            warnings.warn('{0}个指定变量被移除:\n{1}'.format(len(x_except), x_except))
        x = x_inter if x_inter else col_all
    return x


def check_unique_value_prop(df: DataFrame, p: float = .95) -> Dict:
    assert p <= 1, '指定的占比占比需要小于1(100%)'
    assert isinstance(df, DataFrame), '指定的数据集需要是DataFrame'
    mask = [df[col].value_counts().max() / df.shape[0] < p
            for col in df.columns]
    inv_mask = [False if x else True for x in mask]
    removed_columns = df.columns[mask].to_list()
    return {'reserved_data': df.loc[:, mask], 'removed_columns': removed_columns}


def remove_datetime_cols(df: DataFrame) -> DataFrame:
    datetime_cols = df.apply(pd.to_numeric, errors='ignore').select_dtypes(include='object').apply(
        pd.to_datetime, errors='ignore').select_dtypes(include='datetime64').columns.to_list()
    if len(datetime_cols) > 0:
        print(
            f'[HINT] remove {len(datetime_cols)} column(s)\n {datetime_cols}')
        return df.drop(datetime_cols, axis='columns')
    else:
        print(f'[PASS] datetime not found')
        return df


def rep_blank_na(df: DataFrame) -> DataFrame:
    blank_cols = []
    for col in df.columns:
        if df[col].astype(str).str.findall(
                r'^\s*$').apply(lambda x: 1 if len(x) > 0 else 0).sum():
            blank_cols.append(col)
    if len(blank_cols) > 0:
        print(
            f'[HINT] find space in {len(blank_cols)} column(s), replaced by NaN')
        return df.replace(r'^\s*$', np.nan, regex=True)
    else:
        print(f'[PASS] blank not found')
        return df


def check_break_list(breaks: list, features: list):
    if breaks is not None:
        if isinstance(breaks, str):
            breaks = eval(breaks)
        if not isinstance(breaks, dict):
            raise Exception('[Incorrect inputs]')
    pass


def dict_type_check(var, allow_none: bool = True):
    if allow_none:
        if var is not None:
            assert isinstance(var, dict), '[Error] variable must be a dict'
    else:
        assert isinstance(var, dict), '[Error] variable must be a dict'
