"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 10:39:00
"""

import pandas as pd
from data_wrangling import DataWrangling
from pandas import DataFrame


class SCModel(DataWrangling):
    def __init__(self,
                 filename: str,
                 extend: str = 'csv',
                 label: str = 'label',
                 specified_cols: list | None = None):
        self.filename = filename
        self.extend = extend
        self.label = label
        self.specified_cols = specified_cols
        DataWrangling.__init__(self, label=self.label)

    def data_loader(self):
        path = self.filename + '.' + self.extend
        self.data = pd.read_csv(path, encoding='utf-8')
        self.target_exist_na_flg = self.data[self.label].isnull().any()

    def data_view(self):
        self.data_shape(self.data)
        self.target_distribution_check(
            self.data, target_na_flg=self.target_exist_na_flg)
        self.feature_dtypes(self.data)

        self.missing_value_check(self.data)
