"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-20 11:02:38
"""
import pandas as pd
from pandas import DataFrame
from typing import TypeAlias
from dataprocess import DataWrangle
list_check: TypeAlias = list | None


class ScoreCard(DataWrangle):
    def __init__(self,
                 data: DataFrame = None,
                 target: str = 'label',
                 remove_features: list_check = None):
        DataWrangle.__init__(self, data=data, target=target)
        self.data = data
        self.target = target
        assert target in self.data.columns, 'label not in data columns'
        # 所有字段名全部转化为小写
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

        # assert isinstance(
        #     remove_features, list_check), 'remove_features should be a list or None'
        # if remove_features:
        #     for feature in remove_features:
        #         print(feature)
        #         assert feature.lower() in self.features, 'specified removed feature not in data features'
        # print(self.features)

    def data_insight(self):
        self.update_metadata()
        # 数据查看
        self.check_y_distribution()
        # self.check_features_type()
        # self.check_uni_characters()
        pass

    def data_cleaning(self):
        pass


if __name__ == '__main__':
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    sc = ScoreCard(data, 'Churn')
    sc.data_insight()
    print(sc.category_features)
    print(sc.numerical_features)
