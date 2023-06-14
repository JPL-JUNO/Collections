"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 10:39:00
"""

import pandas as pd
from data_loader import DataLoader
from pandas import DataFrame


class SCModel(DataLoader):
    def __init__(self, df: DataFrame, label: str = 'label',
                 specified_cols: list | None = None):
        self.data = df
        self.label = label

        self.specified_cols = specified_cols

    def data_view(self):
        pass
