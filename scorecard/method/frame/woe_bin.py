"""
@Description:
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 15:29:42
"""

import timeit
import platform
import numpy as np
import pandas as pd
import multiprocessing as mp
from pandas import DataFrame
from method.frame.util import check_unique_value_prop
from method.frame.util import remove_datetime_cols
from method.frame.util import rep_blank_na
from method.frame.util import x_variable
from method.frame.util import check_print_step
from method.frame.util import dict_type_check
from method.frame.util import check_special_values
from typing import TypeAlias

Numerical: TypeAlias = int | float


def split_vec_to_df(vec: list) -> DataFrame:
    df = pd.DataFrame({'bin_chr': vec, 'value': vec})
    df = df.reset_index().rename({'index': 'rowid'}, axis='columns')
    df['value'] = df['value'].replace({'missing': np.nan})
    return df


def woe_bin(df, y: str, x: list | None = None, var_skip=None, breaks: dict = None,
            special_values=None, stop_limit: float = .1, count_distr_limit: float = .05, bin_num_limit: int = 8,
            positive: str = 'bad|1', no_cores=None, print_step=0, method='tree', ignore_const_cols=True, ignore_datetime_cols: bool = True,
            replace_blank: bool = True, save_breaks_lst=None, **kwargs):
    start_time = timeit.default_timer()
    # initialize some arguments in **kwargs
    print_info = kwargs.get('print_info', True)
    min_pct_fine_bin = kwargs.get('min_pct_fine_bin', None)
    init_cnt_distr = kwargs.get('init_cnt_distr', min_pct_fine_bin)
    if init_cnt_distr is None:
        init_cnt_distr = .02

    # min_pct_coarse_bin = kwargs.get('min_pct_coarse_bin', None)
    # if min_pct_coarse_bin is not None:
    #     count_distr_limit = min_pct_coarse_bin

    # max_num_bin = kwargs.get('max_num_bin', None)
    # if max_num_bin is not None:
    #     bin_num_limit = max_num_bin
    if print_info:
        print('[INFO] creating WOE binning...')
    df = df.copy(deep=True)
    assert isinstance(y, str), 'y must be a string'
    if x is not None:
        assert isinstance(x, list), 'x must be a list'
        df = df[[y] + x]
    if ignore_const_cols:
        df = check_unique_value_prop(df, p=1)['reversed_data']

    # 删除时间变量 datetime64
    if ignore_datetime_cols:
        df = remove_datetime_cols(df)

    if replace_blank:
        df = rep_blank_na(df)
    features_to_deal = x_variable(df, y, x, var_skip)
    features_num = len(features_to_deal)

    prints_step = check_print_step(print_step)

    breaks = dict_type_check(breaks)

    special_values = check_special_values(special_values, features_to_deal)
    if not isinstance(print_step, int):
        raise ValueError(
            f'[Error] print_step must be non-negative integer, got {print_step}.')
    if print_step < 0:
        print(
            f'[Warning] print_step should be non-negative integer, got {print_step}, change to 1.')
    print_step = 1
    if breaks is not None:
        assert isinstance(
            breaks, dict), '[Error] breaks could be None or Dict'
    if method not in ('tree', 'chimerge'):
        print(
            f'[Warning] method must be one of tree or chimerge, got {method}, will be set to "tree"')
        method = 'tree'
    if (no_cores is None) or (no_cores < 1):
        all_cores = mp.cpu_count() - 1
        no_cores = int(np.ceil(features_num / 5 if features_num /
                       5 < all_cores else all_cores * .9))
    if platform.system() == 'Windows':
        no_cores = 1

    if no_cores == 1:
        bins = {}
        for i in range(features_num):
            x_i = features_to_deal[i]
            if print_step > 0 and bool((i + 1) % print_step):
                print(f'{i+1}/{features_num} {x_i}')
            bins[x_i] = woe_bin_univariate(
                df=pd.DataFrame(
                    {'y': df[y], 'variable': x_i, 'value': df[x_i]}),
                breaks=breaks[x_i] if (breaks is not None) and (
                    x_i in breaks.keys()) else None,
                special_val=special_values[x_i] if (special_values is not None) and (
                    x_i in special_values.keys()) else None,
                init_cnt_distr=init_cnt_distr,
                cnt_distr_limit=count_distr_limit,
                stop_limit=stop_limit,
                bin_num_limit=bin_num_limit,
                method=method
            )
    else:
        pool = mp.Pool(processes=no_cores)
    pass


def woe_bin_univariate(df: DataFrame, breaks: list | None = None, special_val=None,
                       init_cnt_distr: float = .02, cnt_distr_limit: float = .05,
                       stop_limit: float = .1, bin_num_limit: int = 8, method: str = 'tree'):
    if breaks is not None:
        bin_list = woe_bin_2_break(
            df=df, breaks=breaks, special_val=special_val)
    else:
        if stop_limit == 'N':
            bin_list = woe_bin_univariate_init_bin(df)
        else:
            if method == 'tree':
                bin_list = woe_bin_univariate_tree()
            elif method == 'chimerge':
                bin_list = woe_bin_univariate_chimerge()


def woe_bin_univariate_init_bin(df, init_cnt_distr, breaks, special_val):

    pass


def woe_bin_univariate_tree():
    pass


def woe_bin_univariate_chimerge():
    pass


def df_binning_save(df, breaks, special_val) -> dict:
    special_val = add_missing_special_values(df, breaks, special_val)
    if special_val is not None:
        special_val_df = split_vec_to_df(special_val)

    return {}


def woe_bin_2_break(df: DataFrame, breaks: list, special_val):
    break_df = split_vec_to_df(breaks)
    df_binning_save(df, breaks, special_val)
    pass


def add_missing_special_values(df, breaks, special_val):
    if df['value'].isnull().any():
        if breaks is None:
            return contain_missing_str(special_val)
        elif any(['missing' in str(i) for i in breaks]):
            return special_val
        else:
            return contain_missing_str(special_val)


def contain_missing_str(special_value: list) -> list:
    """判断一个字符串列表是否存在某一项包含missing，没有则添加，有则不操作

    Args:
        special_value (list): 待判断的列表

    Returns:
        list: 包含missing的列表
    """
    if special_value is None:
        return ['missing']
    elif any(['missing' in str(i) for i in special_value]):
        return special_value
    else:
        return ['missing'] + special_value


def num_0(ser) -> int:
    return sum(ser == 0)


def num_1(ser) -> int:
    return sum(ser == 1)
