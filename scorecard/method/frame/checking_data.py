"""
@Description:
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-05 10:15:19
"""

import time
import inspect
import warnings
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from collections import Counter
from method.frame.features_derive import Derive
from method.frame.standard_scaler import StandardScaler
from method.temp.var_stat import vb_code


class DataMining(Derive, StandardScaler):
    def __init__(self, data: DataFrame, label: str = 'label'):
        self.step = 0
        self.data = data
        self.label = label

        self.columns = self.data.columns
        self.shape = self.data.shape
        self.m, self.n = self.data.shape
        self.dtypes = self.data.dtypes

        self.print_lvl = 99

        if isinstance(vb_code, dict):
            try:
                self.vb_code = vb_code
            except NameError:
                warnings.warn(' No variable coding determined...')
                self.vb_code = dict()
        else:
            self.vb_code = dict()

    def check_y_dist(self):
        self._print_step('检查样本分布')
        a = Counter(self.data[self.label])
        for k, v in a.items():
            self._print(
                'Label {0:^4} has {1:>8} samples {2:2.2%}.'.format(k, v, v / self.m))
        print('')

    def _print_step(self, info: str):
        s = inspect.stack()[1][3]
        print('Step {} {} {}...'.format(self.step, info, s))
        self.step += 1

    def _print(self, p: str):
        if self.print_lvl > 3:
            print(p)

    def check_dtypes(self):
        self._print_step('检查特征类型')
        for k, v in Counter(self.data.dtypes).items():
            self._print(
                'Data type {0} has {1} feature(s) proportion {2:2.2%}'.format(k, v, v / self.n))
        print('')

    def check_uni_char(self, uni):
        self._print_step('清理异常字符')
        # 获取object的字段名
        # cols = pd.DataFrame(self.data.dtypes)
        # cols = list(cols.loc[cols[0] == 'object'].index)
        cols = self.data.select_dtypes(include='object').columns.to_list()

        x = 0
        for col in cols:
            self.data[col] = self.data[col].fillna('')
            try:
                self.data[col] = self.data[col].str.replace(uni, '')
            except:
                pass
            x += 1
        self._print('在 {1} 个非数值型字段清理字符串 {0}'.format(uni, x))
        print('')

    def check_missing_value(self, print_result: bool = False):
        self._print_step('缺失值检查')

        if self.vb_code is None:
            self.vb_code = dict()
        tem = self.data.isna().sum()
        check_mv = pd.DataFrame(
            round(tem[tem > 0] / self.m, 3), columns=['nan_#'])
        # tem = self.data.columns[self.data.isna().sum() > 0].to_list()
        lst = list()
        for i in check_mv.index:
            try:
                lst.append(self.vb_code[i])
            except:
                lst.append('-')
        check_mv['dtypes'] = self.data.dtypes[check_mv.index]
        check_mv['vb'] = lst

        # check_mv = check_mv.assign(
        #     dtypes=lambda x: [self.data.dtypes[x] for x in check_mv.index])
        self.na_summary = check_mv

        if print_result:
            self._print(' 特征缺失率：')
            self._print(self.na_summary)
            print('')

    def filter_abnor_values(self, abnor):
        assert isinstance(
            abnor, list), 'The declaration of abnor must be in list'
        for i in abnor:
            assert len(i) == 4, 'The length of rule in declaration must be 4!'
        for i in abnor:
            var, ab, target, sig = i
            if var in self.data.columns:
                self._filter_abnor_values0(var, ab, target, sig)

    def _filter_abnor_values0(self, var, ab, target, sig):
        n = eval('len(self.data.loc[self.data[var] {} ab, var])'.format(sig))
        r = n / self.m
        exec('self.data.loc[self.data[var] {} ab, var]=target'.format(sig))
        self._print(' 特征 {0:,12} 异常值处理 替换 {1} {2:>9} 为 {:>5,} 影响 {:>7,} {:2.1%} 样本...'.format(
            var, sig, ab, target, n, r))

    def filter_blank_values(self) -> None:
        self._print_step('填充空白字符串')
        self.data = self.data.replace(r'^\s*$', np.nan, regex=True)

    def fill_missing_values(self, fill, print_step: bool = True) -> None:
        pass
