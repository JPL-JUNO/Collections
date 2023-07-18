"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-17 09:30:27
"""
from pandas import DataFrame
from timeit import default_timer
from typing import TypeAlias
from collections import defaultdict
numerical: TypeAlias = float | int
import numpy as np
from pandas.api.types import is_numeric_dtype


def woebin_tree(df):
    pass


def woebin_chi2merge(df: DataFrame, stop_limit: float = .1):
    def add_chisq(initial_binning):
        pass
    bin_list = woebin2_init_bin()
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    if len(initial_binning.index) == 1:
        return {'bining_sv': binning_sv, 'binning': initial_binning}
    df_rows = len(df)
    from scipy.special import chdtri
    chisq_limit = chdtri(1, stop_limit)
    binning_chisq = add_chisq(initial_binning)

    pass


def woebin2_init_bin(dtm, init_cnt_distr, breaks, spl_val):
    dtm, binning_sv = dtm_binning_sv(dtm, breaks, spl_val)
    if dtm is None:
        return binning_sv, None
    if is_numeric_dtype(dtm['value']):
        xvalue = dtm['value'].astype(float)
        iq = xvalue.quantile([.01, .25, .75, .99])
        iqr = iq.loc[.75] - iq.loc[.25]
        if iqr == 0:
            prob_down = .01
            prob_up = .99
        else:
            prob_down = .25
            prob_up = .75
        xvalue_rm_outlier = xvalue[(
            xvalue >= iq.loc[prob_down] - 3 * iqr) & (xvalue <= iq[prob_up] + 3 * iqr)]
        n = np.trunc(1 / init_cnt_distr)
        len_uniq_x = len(np.unique(xvalue_rm_outlier))
        if len_uniq_x < n:
            n = len_uniq_x
        brk = np.unique(xvalue_rm_outlier) if len_uniq_x < 10 else pretty(
            min(xvalue_rm_outlier), max(xvalue_rm_outlier), n)
        brk = list(filter(lambda x: x > np.nanmin(
            xvalue) and x <= np.nanmax(xvalue), brk))
    pass


def pretty():
    pass


def split_vec_todf():
    pass


def dtm_binning_sv(dtm, breaks, spl_val) -> tuple:
    # spl_val = add_missing_spl_val(dtm, breaks, spl_val)
    if spl_val is not None:
        sv_df = split_vec_todf(spl_val)
        if is_numeric_dtype(dtm['value']):
            sv_df['value'] = sv_df['value'].astype(dtm['value'].dtypes)
            sv_df['bin_chr'] = np.where(np.isnan(
                sv_df['value']), sv_df['bin_chr'], sv_df['value'].astype(dtm['value'].dtypes).astype(str))
        dtm_sv = pd.merge(dtm.fillna('missing'), sv_df[['value']].fillna(
            'missing'), how='inner', on='value', right_index=True)
        # 将原本的dtm分为两个新的dataframe
        # dtm_sv 包含所有的 spl_val
        # dtm 剔除所有的 spl_val
        dtm = dtm[~dtm.index.isin(dtm_sv.index)].reset_index() if len(
            dtm_sv) < len(dtm) else None
        if dtm_sv.shape[0] == 0:
            return {'binning_sv': None, 'dtm': dtm}
        binning_sv = pd.merge(dtm_sv.fillna('missing').groupby(
            ['variable', 'value'])['y'].agg([n0, n1]).reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}),
            sv_df.fillna('missing'), on='value').groupby(['variable', 'rowid', 'bin_chr']).agg({'bad': sum, 'good': sum}).reset_index().rename(columns={'bin_chr': 'bin'}).drop('rowid', axis=1)
    else:
        binning_sv = None
    return dtm, binning_sv


def n0(x):
    return sum(x == 'Iris-setosa')


def n1(x):
    return sum(x == 'Iris-virginica')


def woebin2_breaks():
    pass


def woebin2(df: DataFrame, breaks, spl_val=None,
            init_cnt_distr: float = .02,
            cnt_distr_limit: float = .05, stop_limit: float = .1,
            bin_num_limit: int = 8, method: str = 'tree'):
    if breaks is not None:
        bins = woebin2_breaks()
    else:
        if method == 'tree':
            bins = woebin_tree()
        else:
            bins = woebin_chi2merge()
    pass


def woebin(df: DataFrame, y: str = 'label', x=None,
           var_skip=None, breaks_list=None, special_values=None,
           stop_limit: numerical = .1,
           method: str = 'tree'):
    start_time = default_timer()
    print('[信息] 开始进行WOE分箱')
    if x is not None:
        # 如果x不是空的话，即有传入变量，则表示使用x+y来作为建模的变量，否则默认df中全部字段
        df = df[[y] + x]

    if stop_limit not in pd.Interval(left=0, right=.5, closed='both'):
        raise ValueError(f'stop_limit must be in [.0, .5], got {stop_limit}')

    if method not in ('tree', 'chimerge'):
        raise ValueError(
            f"method should be 'tree' or 'chimerge', got {method}")

    # 因为df中包含y,因此在循环特征时，剔除y
    bins = defaultdict(list)
    for xi in df.columns.drop(y):
        df_immd = pd.DataFrame({'y': df[y], 'variable': xi, 'value': df[xi]})
        bins[xi] = woebin2()
        if breaks_list is not None:
            pass
        else:
            if method == 'tree':
                bin_list = woebin_tree()
            else:
                bin_list = woebin_chi2merge()
    end_time = default_timer()
    print(f'[信息] 运行时间：{(end_time-start_time):.4f}s')


if __name__ == '__main__':
    # import pandas as pd
    # df = pd.read_csv('tidy.data', names=['a', 'b', 'c', 'd', 'label'])
    # iv = pd.Interval(left=0, right=.5, closed='both')
    # assert .3 in iv, 'add'
    # woebin(df)
    import pandas as pd
    df = pd.read_csv('tidy.data', names=['A', 'B', 'C', 'D', 'label'])
