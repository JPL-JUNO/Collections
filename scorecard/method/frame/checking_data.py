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
from method.frame.util import feature_zip
from method.temp.var_stat import vb_code


class DataMining(Derive, StandardScaler):
    def __init__(self, data: DataFrame, label: str = 'label'):
        self.step = 1
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

    def renew(self, pnt: bool = True, desc: bool = True) -> None:
        if desc:
            old_shape = self.shape
        self.columns = self.data.columns
        self.shape = self.data.shape
        self.m, self.n = self.data.shape

        if desc:
            delta_col = self.n - old_shape[1]
            delta_row = self.m - old_shape[0]
            if delta_col < 0:
                info_col, delta_col = '特征数减少', -delta_col
            elif delta_col > 0:
                info_col, delta_col = '特征数增加', delta_col
            else:
                info_col, delta_col = '特征数未变', ''

            if delta_row < 0:
                info_row, delta_row = '样本数减少', -delta_row
            elif delta_col > 0:
                info_row, delta_row = '样本数增加', delta_row
            else:
                info_row, delta_row = '样本数未变', ''
            self._print('{0:<6}{1} {2:<6}{3}'.format(
                info_col, delta_col, info_row, delta_row))
        if pnt:
            self._print(
                '数据更新：\n最新样本数为{0:,} 最新特征数为{1:,}'.format(self.m, self.n))
            print('')

    def _print_step(self, info: str) -> None:
        s = inspect.stack()[1][3]
        print('Step {} {} {}...'.format(self.step, info, s))
        self.step += 1

    def _print(self, p: str) -> None:
        if self.print_lvl > 3:
            print(p)

    def check_dtypes(self) -> None:
        self._print_step('检查特征类型')
        for k, v in Counter(self.data.dtypes).items():
            self._print(
                'Data type {0} has {1} feature(s) proportion {2:2.2%}'.format(k, v, v / self.n))
        print('')

    def check_uni_char(self, uni) -> None:
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
            self._print('特征缺失率：')
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
        print('')

    def fill_missing_values(self, mapping: dict, print_step: bool = True) -> None:
        self._print_step('填充缺失数据')
        for k, v in mapping.items():
            na_num = self.data[k].isnull().sum()
            self.data[k] = self.data[k].fillna(v)
            if print_step:
                self._print('特征{0}填充{1:>6}个缺失值{2:>4}'.format(k, na_num, v))
        print('')

    def filter_data_subtable(self, frac=None,
                             balance: bool = True,
                             oversampling: bool = False,
                             label: str = 'label', random_state=42) -> None:
        self._print_step('不平衡样本重构')
        a = Counter(self.data[self.label])
        k_more, v_more = a.most_common(2)[0]
        k_less, v_less = a.most_common(2)[1]
        if balance:
            if oversampling:  # 上采样
                self.data = pd.concat(
                    [self.data.loc[self.data[label] == k_more],
                     self.data.loc[self.data[label] == k_less].sample(
                        frac=v_more / v_less, random_state=random_state, replace=True).sort_index()
                     ], ignore_index=True
                )
            else:
                self.data = pd.concat(
                    # 无放回抽样
                    [self.data.loc[self.data[label] == k_more].sample(
                        frac=v_less / v_more, random_state=random_state, replace=True).sort_index(),
                     self.data.loc[self.data[label] == k_less]
                     ], ignore_index=True
                )
                self.renew()
                self.check_y_dist()
        else:
            # 如果frac不是整数或者小数，抛出错误
            assert isinstance(frac, (float, int)), 'frac not right determined.'
            self.data = pd.concat(
                [self.data.loc[self.data[label] == k_more].sample(
                    frac=frac, random_state=random_state).sort_index(),
                 self.data.loc[self.data[label] == k_less]
                 ], ignore_index=True
            )
            self.renew()
            self.check_y_dist()

    def data_describe(self) -> DataFrame:
        self._print('更新特征描述')
        epo = pd.DataFrame(index=list(self.columns))
        epo['dtypes'] = self.data.dtypes[epo.index]
        epo['vb_name'] = [self.vb_code[i]
                          if i in self.vb_code.keys() else '-' for i in epo.index]
        epo['total'] = self.m
        epo['identical_value'] = [
            self.data[col].value_counts(normalize=True).max() for col in epo.index]
        epo = pd.merge(epo, self.data.describe(percentiles=[.05, .25, .75, .95]).T,
                       how='left', left_index=True, right_index=True)
        epo.drop(['count'], axis=1, inplace=True)
        epo = epo.rename(columns={'total': 'count'})
        order = ['dtypes', 'vb_name', 'count', 'mean', 'std', 'min',
                 '5%', '25%', '50%', '75%', '95%', 'max', 'identical_value']
        epo = epo[order]
        self.epo = epo
        return self.epo

    def check_feature_zip(self, var: dict, c: float = .3, if0: bool = False, plot: bool = False) -> None:
        self._print_step('连续特征压缩')

        # 目标特征存在于 self.data 的字段中
        self.__k_in = list(set(var.keys()).intersection(self.columns))
        # 不存在数据字段中的目标特征
        self.__k_out = list(set(var.keys()).difference(set(self.__k_in)))
        # 备份传入特征的数据以及target
        self.__date_feature_zip_backup = self.data[self.__k_in + [
            self.label]].copy(deep=True)
        for k in self.__k_in:
            feature_zip(self.__date_feature_zip_backup, var=k, c=c,
                        e=var[k], if0=if0, inplace=True, label=self.label, plot=plot)
        for k in self.__k_out:
            self._print('特征{0}不存在...'.format(k))
        pass

    def copy_filter_feature_zip(self) -> None:
        self._print_step('压缩特征-测试数据')
        self.test_data = self.data.copy(deep=True)
        self.test_data.drop(self.__k_in, axis=1, inplace=True)
        self.__date_feature_zip_backup.drop(self.label, axis=1, inplace=True)
        self.test_data = pd.concat(
            [self.test_date, self.__date_feature_zip_backup], axis=1)
        self.renew()

    def sample_var_filter(self, dt: DataFrame, x=None,
                          iv_limit: float = .02, missing_limit: float = .95, identical_limit: float = .95,
                          var_rm: list | None = None, var_kp: list | None = None,
                          return_rm_reason: bool = True, positive: bool = True) -> Series:
        self._print_step('特征过滤')
        tem = var_filter()
        if return_rm_reason:
            self.rm_reason = tem['rm']
        return tem['dt']
