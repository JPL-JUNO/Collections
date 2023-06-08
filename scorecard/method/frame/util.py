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


def check_y(data, y: str, positive):
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
