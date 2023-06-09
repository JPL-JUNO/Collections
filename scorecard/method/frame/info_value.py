"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-09 16:31:47
"""

import pandas as pd
import numbers as np
from method.frame.util import check_y, x_variable
from pandas import DataFrame, Series


def information_value(df: DataFrame, y: str, x: list | None = None,
                      positive='bad|1',
                      order: bool = True):
    df = df.copy(deep=True)
    if x is not None:
        assert isinstance(x, list), 'x should be a list'
    assert isinstance(y, str), 'y should be a string'
    df = df[[y] + x]
    df = check_y(df, y, positive)
    cols = x_variable(df, y, x)
    iv_lst = pd.DataFrame({
        'variable': cols,
        'IV': [iv_calc(df[xi], df[y[0]]) for xi in cols]
    })
    pass


def iv_calc(x: Series, y: Series):
    pass


df.pivot_table()
