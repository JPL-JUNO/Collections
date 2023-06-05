"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-05 10:26:13
"""
import numpy as np
import pandas as pd


class ReadData(object):
    def __init__(self, route: str):
        self.tem = None
        self.data = None
        self.route = route
        self.rm = []

    def read_table(self, encoding: str = 'utf-8', sep: str = ','):
        self.data = pd.read_csv(
            self.route, encoding=encoding, sep=sep, low_memory=True)
        for col in self.rm:
            if col in self.data.columns:
                self.data.drop(col, axis=1, inplace=True)
        return self.data
