"""
@Description: Implement DataLoader to read data
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-14 11:04:52
"""
import pandas as pd


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def read_data(self):
        self.data = pd.read_csv(self.path, encoding='utf-8')
