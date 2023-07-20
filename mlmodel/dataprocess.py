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
        pass

    def check_y_distribution(self):
        na_cnt = self.target_data.isnull().sum()
        if na_cnt:
            print(f'[提醒] target 中存在缺失值，缺失数量为 {na_cnt}')
        else:
            print(f'[信息] target 中不存在缺失值')
        print(f'[信息] target 的分布情况：\n{self.target_data.value_counts()}')

        self.target_na_flag = True if na_cnt else False

    def check_features_type(self, data: DataFrame) -> None:
        print(f'[信息] features 的数据类型\n{data.dtypes.value_counts()}')

    def metadata(self, data: DataFrame) -> None:
        dtypes = data.dtypes.value_counts().index
        pass

    def update_metadata(self):
        print(self.data.shape)
        pass

    def check_uni_characters(self, data: DataFrame, uni: str) -> DataFrame:
        assert isinstance(uni, str), 'uni should be a list or None'
        changed_col = 0
        for col in data.columns:
            data[col] = data[col].fillna('')
            if data[col].str.contains(uni, regex=False).any(axis=0):
                data[col] = data[col].str.replace(uni, '')
                changed_col += 1
        print(f'[信息] {uni} 出现在 {changed_col} 个分类特征中，已经替换')
        return data
