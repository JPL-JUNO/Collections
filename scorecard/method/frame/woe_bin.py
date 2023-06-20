"""
@Description:
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 15:29:42
"""

import timeit
from frame.util import check_unique_value_prop
from frame.util import remove_datetime_cols
from frame.util import rep_blank_na
from frame.util import x_variable


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
    pass
