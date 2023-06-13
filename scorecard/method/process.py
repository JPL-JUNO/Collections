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
                              list), 'Specified columns must be in a list'
            self.data = self.data[[
                self.label] + self.use_specified_col]
            self.renew()
        # target的分类统计结果并打印
        self.check_y_dist()

        # 检查数据类型
        self.check_dtypes()

        # 异常字符串处理
        if cek_uni_char is not None:
            for i in cek_uni_char:
                self.check_uni_char(i)

        if fillna is not None:
            self.fill_missing_values(mapping=fillna)
        # 移除异常值
        if abnor is not None:
            self.filter_abnor_values(abnor)

        if remove_blank:
            self.filter_blank_values()
        # 缺失值检查
        self.check_missing_value(print_result=True)
        # 样本平衡
        if resample:
            self.filter_data_subtable(
                label=self.label, balance=True, oversampling=oversampling)
        # 最终的样本描述
        self.check_y_dist()

        self.epo = self.data_describe()

    def pro_feature_filter(self, inplace_data: bool = True,
                           var_zip=None,
                           plot_zip: bool = False,
                           iv_limit: float = .02,
                           missing_limit: float = .95,
                           identical_limit: float = .95,
                           var_rm: list = None,
                           var_kp: list = None,
                           positive: str = 'good|1'):
        if var_zip is None:
            numerical_col = self.data.drop(self.label, axis=1).select_dtypes(
                include=['int', 'float']).columns
            var_zip = {col: None for col in numerical_col}
        if var_kp is None:
            var_kp = list()
        var_kp2 = list() if self.use_specified_col is None else self.use_specified_col
        self.check_feature_zip(var_zip, c=.3, if0=False, plot=plot_zip)

        # 创建 test_data
        self.copy_filter_feature_zip()

        self.test_data = self.sample_var_filter(dt=self.test_data,
                                                x=None,
                                                iv_limit=iv_limit,
                                                missing_limit=missing_limit,
                                                identical_limit=identical_limit, var_rm=var_rm,
                                                var_kp=list(
                                                    set(var_kp + var_kp2)),
                                                return_rm_reason=True, positive=positive)
        if inplace_data:
            self.data = self.data[self.test_data.columns.tolist()]
            self.renew()
        self.epo = self.data_describe()

    def feature_process(self, iv_threshold: float = .15,
                        max_features: int = 6,
                        corr_threshold: float = .6,
                        cum_importance: float = .95,
                        breaks_adj=None, var_remove: list | None = None,
                        var_keep: list | None = None) -> None:
        if var_keep is not None:
            assert isinstance(var_keep, list), 'var_keep must be a list'
        if var_remove is not None:
            assert isinstance(var_remove, list), 'var_remove must be a list'
        if self.use_specified_col is not None:
            assert isinstance(self.use_specified_col,
                              list), 'use_specified_col must be a list'
            print('使用指定特征建模...')
            self.data = self.data[[self.label] + self.use_specified_col]
            self.renew()
        else:
            self.bin0 = self.sample_woe_bin()
