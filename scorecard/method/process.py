import pandas as pd
from method.frame.checking_data import DataMining


class ScoreCardProcess(DataMining):
    def __init__(self, data,
                 label: str = 'label',
                 show_plot: bool = False):
        self.data = data
        self.label = label
        self.show_plot = show_plot
        self.use_specified_col = None

        DataMining.__init__(self, self.data, self.label)

    def pro_check_data(self, fillna: dict = None,
                       abnor: list = None,
                       remove_blank: bool = True,
                       resample: bool = True,
                       oversampling: bool = False,
                       cek_uni_char: list = ["'", ""]):
        # 使用指定特征建模
        if self.use_specified_col is not None:
            assert isinstance(self.use_specified_col,
                              list), 'Specified columns should be in a list'
            self.data = self.data = self.data[[
                self.label] + self.use_specified_col]
        # target的分类统计结果并打印
        self.check_y_dist()

        # 检查数据类型
        self.check_dtypes()
        # 异常字符串处理
        if cek_uni_char is not None:
            for i in cek_uni_char:
                self.check_uni_char(i)

        if fillna is not None:
            self.filter_missing_values(fillna)

        if abnor is not None:
            self.filter_abnor_values(abnor)

        if remove_blank:
            self.filter_blank_values()
        self.check_missing_value(print_result=True)
