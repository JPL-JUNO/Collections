"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-20 14:21:39
"""
import pandas as pd
from pandas import Series, DataFrame
from typing import TypeAlias
str_check: TypeAlias = str | None


class DataWrangle:
    def __init__(self, data: DataFrame,
                 target: str = 'label') -> None:
        # self.target = target
        # 这里写只是为了写代码时候的提示
        self.data = data
        self.target = target
        self.data.columns = self.data.columns.str.replace(' ', '_').str.lower()
        # target也转为小写形式
        self.target = target.lower()
        self.target_data = self.data[self.target]
        # 获取数据中的所有特征
        self.features = self.data.columns.drop(self.target)
        self.features_data = self.data[self.features]

        self.category_features = self.features[self.features_data.dtypes == 'object']
        self.numerical_features = self.features_data.select_dtypes(
            include='number').columns
        pass

    def check_y_distribution(self, drop_target_na: bool = True):
        na_cnt = self.target_data.isnull().sum()
        if na_cnt:
            print(f'[提醒] target 中存在缺失值，缺失数量为 {na_cnt}')
            if drop_target_na:
                self.data = self.data.dropna()
        else:
            print(f'[信息] target 中不存在缺失值')
        print(f'[信息] target 的分布情况：\n{self.target_data.value_counts()}')

        self.target_na_flag = True if na_cnt else False

    def check_features_type(self) -> None:
        print(
            f'[信息] features 的数据类型\n{self.features_data.dtypes.value_counts()}')

    def metadata(self, data: DataFrame) -> None:
        dtypes = data.dtypes.value_counts().index
        pass

    def update_metadata(self):
        print(self.data.shape)
        pass

    def check_uni_characters(self, uni: str) -> DataFrame:
        assert isinstance(uni, str), 'uni should be a list or None'
        changed_col = 0
        for col in self.category_features:
            self.data[col] = self.data[col].fillna('')
            if self.data[col].str.contains(uni, regex=False).any(axis=0):
                self.data[col] = self.data[col].str.replace(uni, '')
                changed_col += 1
        print(f'[信息] {uni} 出现在 {changed_col} 个分类特征中，已经替换')

    def check_na(self, fillna: dict = None):
        self.feature_na_flag = self.features_data.isnull().any(axis=0).any(axis=0)
        if self.feature_na_flag:
            if fillna == 0:
                pass
        else:
            print('[信息] feature 中不存在缺失值')
