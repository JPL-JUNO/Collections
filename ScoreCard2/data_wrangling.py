"""
@Description: Data wrangling methods may be used in data preprocessing 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-15 14:56:46
"""
import pandas as pd
from pandas import DataFrame
import inspect
from collections import Counter
from woe import woebin


class DataWrangling:
    def __init__(self, label: str = 'label'):
        self.processing_step = 1
        self.label = label

    def __hint_info_print(self, info: str) -> None:
        s = inspect.stack()[1][3]
        print(
            f'Precessing Info[Step {self.processing_step:2d}] {info} [{s}]...')
        self.processing_step += 1

    def data_shape(self, data: DataFrame):
        self.__hint_info_print('data shape')
        print(
            f'DataFrame has shape {data.shape[1]-1} feature(s) and {data.shape[0]} sample(s)')
        print('')

    def target_distribution_check(self, data: DataFrame, target_na_flg: bool = False):
        self.__hint_info_print('distribution of target class')
        if target_na_flg:
            print('Warning: target exists missing value')
        m = data.shape[0]
        cnt = Counter(data[self.label])
        for k, v in cnt.items():
            print(f'Label {k} has {v} samples proportion to {100*v/m:.2f}%')
        print('')

    def feature_dtypes(self, data: DataFrame):
        self.__hint_info_print('data type of features')
        cnt = Counter(data.drop(self.label, axis=1).dtypes)
        n = data.shape[1] - 1
        for k, v in cnt.items():
            print(f'{k} has {v:2d} feature(s) proportion to {100*v/n:.2f}%')
        print('')

    def missing_value_check(self, data: DataFrame):
        self.__hint_info_print('missing value check of feature and target')
        self.na_stat = data.isnull().sum() / data.shape[0]
        feature_na_stat = self.na_stat.drop(self.label, axis=0)

        for idx, val in zip(feature_na_stat.index, feature_na_stat.values):
            print(f'{100*val:.3f} % missing rate of feature {idx}')
        if self.na_stat[self.label]:
            print(
                f'{100*self.na_stat[self.label]:.3f} % missing rate of target')
        print('')

    def sample_woebin(self, ):
        print('特征分箱')

        bins = woebin(df=self.data, y=self.label, var_skip=var_skip)
