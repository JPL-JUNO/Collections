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


def woebin_tree(df):
    pass


def woebin_chi2merge(df: DataFrame):
    pass


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
    import pandas as pd
    df = pd.read_csv('tidy.data', names=['a', 'b', 'c', 'd', 'label'])
    iv = pd.Interval(left=0, right=.5, closed='both')
    assert .3 in iv, 'add'
    woebin(df)
