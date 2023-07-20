"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 10:39:00
"""

import pandas as pd
from data_wrangling import DataWrangling
from handling import FeatureSelector
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from typing import TypeAlias
list_check: TypeAlias = list | None


class SCModel(DataWrangling):
    def __init__(self,
                 filename: str,
                 extend: str = 'csv',
                 label: str = 'label',
                 specified_cols: list_check = None):
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

    def pro_feature_process(self, iv_threshold: float = .15,
                            max_feature: int = 6,
                            corr_threshold: float = .6,
                            cum_importance: float = .95,
                            break_adj=None,
                            var_rm: list = None, var_kp=None):
        if not isinstance(var_rm, list_check):
            raise TypeError('var_rm should be a list or None')

        if not isinstance(var_kp, list_check):
            raise TypeError('var_kp should be a list or None')
        if not isinstance(self.specified_cols, list_check):
            raise TypeError('use_specified_var should be a list or None')
        if self.specified_cols is not None:
            print('[信息] 使用指定的特征建模...')
            self.data = self.data[[self.label] + self.specified_cols]
        else:
            self.bins0 = self.sample_woebin(break_list=break_adj,
                                            set_default_bin=False,
                                            no_cores=1)
            self.filter_feature_iv(
                self.bins0, iv=iv_threshold, remove=True, re=False)
            self.check_feature_importances(self.bins0, n_estimators=100,
                                           max_features=max_feature,
                                           max_depth=3)
            self.plot_feature_importances()
            selector = FeatureSelector(
                data=self.data, labels=self.data[self.label])
            # selector.identify_collinear(corr_threshold=.6)
            selector.identify_collinear(corr_threshold=corr_threshold)
            selector.plot_collinear(plot_all=True)
            self.corr_matrix = selector.corr_matrix

            self.check_corr_matrix_control(threshold=corr_threshold, remove=True,
                                           re=False, method='feature_importance')
            self.renew()
            self.check_feature_importances(self.bins0, n_estimators=100,
                                           max_features=max_feature, max_depth=3)
            self.plot_feature_importances()

            self.filter_feature_importances(cum=cum_importance,
                                            method='cum')
            self.renew()
        self.sample_woebin(break_list=break_adj,
                           set_default_bin=True, re=False, no_cores=1)

        pass

    def pro_sampling(self):
        self.sample_split(ratio=.7, seed=123)
        self.sample_woe_ply(self.bins)

    def pro_modeling(self, penalty='l2',
                     C=1, solver: str = 'lbfgs',
                     n_jobs: int = -1):
        self.model = LogisticRegression(penalty=penalty,
                                        C=C,
                                        solver=solver, n_jobs=n_jobs)
        self.model.fit(self.X_train, self.y_train)
        self.train_pred = self.model.predict_proba(self.X_train)[:, 1]
        self.test_pred = self.model.predict_proba(self.X_test)[:, 1]

    def pro_evaluation(self):
        self.train_perf = perf_eva(
            self.y_train, self.train_pred, title='train')
        self.test_perf = perf_eva(self.y_test, self.test_pred, title='test')
        self.model_scorecard()
        self.train_score = self.model_scorecard_ply(self.train, self.card)
        self.test_score = self.model_scorecard_ply(self.test, self.card)

        self.psi = perf_psi(score={'train': self.train_score, 'test': self.test_score},
                            label={'train': self.y_train, 'test': self.y_test},
                            return_distr_dat=True, figsize=(11, 6), show_plot=self.show_plot)

    def pro_development(self, save=True, route='./temp/source', name='output'):
        self.output = defaultdict(dict)
        self.output['KS']['train'] = self.train_perf['KS']
        self.output['KS']['test'] = self.test_perf['test']
        self.output['AUC']['train'] = self.train_perf['AUC']
        self.output['AUC']['test'] = self.test_perf['test']

        print(f"Train set eva: KS = {self.output['KS']['train']}")
        print(f"Test set eva: KS = {self.output['KS']['test']}")
        print(f"Train set eva: AUC = {self.output['AUC']['train']}")
        print(f"Test set eva: AUC = {self.output['AUC']['test']}")
        self.output['psi'] = self.psi['dat']['score']
        self.output['epc'] = self.epo
        self.output['card'] = self.model_card_save()
        if getattr(self, 'bins0', None) is not None:
            pass

        # ls = list(self.bins.values())
        bins_tem = pd.DataFrame(self.bins.values()).reset_index(
            drop=True).sort_values(by=['total_iv', 'variable'], ascending=[False, True])
        bins_tem_coef = pd.DataFrame(
            {'variable': [i.replace('_woe', '') for i in self.X_train.columns],
             'coef': list(self.model.coef_[0]),
             'VIF': [variance_inflation_factor(self.X_train.values, i) for i in range(self.X_train.shape[1])]})
        bins_tem = bins_tem.merge(bins_tem_coef, on='variable', how='left')
        id0 = list()
        for i in bins_tem['variable']:
            if i in vb_code.keys():
                id0.append(vb_code[i])
            else:
                id0.append(i)
        bins_tem['name'] = id0
        bins_tem = bins_tem[['variable', 'name', 'bin', 'count', 'count_distr', 'good',
                             'bad', 'bardprob', 'woe', 'bin_iv', 'total_iv', 'coef', 'VIF']]
        self.output['bins'] = bins_tem
        if save:
            variable_dum(self.output, route=route, name=name)
        xlsx_save = xlsxwriter(filename='output')
        for k, v in self.output.items():
            if isinstance(v, DataFrame):
                if k in ('binsall', 'bins'):
                    comment = ['FEATURE PROJECT BIN', 'WOE']
                    conditional_format = ['count_distr', 'badprob']
                elif k in ('card'):
                    comment = None
                    conditional_format = ['points']
                elif k in ('epo', 'psi'):
                    comment = None
                    conditional_format = None
                else:
                    comment = None
                    conditional_format = None
                xlsx_save.write(data[v],
                                sheet_name=k, startrow=2, startcol=2, index=0,
                                conditional_format=conditional_format, comment=comment)

        xlsx_save.save()
        chartbin = xlsxwriter(filename='chart_bins')
        chartbin.chart_woebin(self.output['binsall'],
                              vb_code,
                              series_name={'good': '负样本',
                                           'bad': '正样本',
                                           'badprop': '正样本占比'})
        chartbin.save()
