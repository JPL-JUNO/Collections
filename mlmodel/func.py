"""
@Description: 提供一些单独的函数支持
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-19 11:26:38
"""

from typing import TypeAlias
numerical: TypeAlias = float | int
import numpy as np
from numpy import ndarray


def pretty(low: numerical, high: numerical, n: int) -> ndarray:
    """实现`pretty`的区间分割，与R中`pretty`函数相同，生成的序列最大值大于等于`high`，最小值小于等于`low`，间隔是10^t的1倍、2倍、5倍

    Parameters
    ----------
    low : numerical
        指定的序列下界
    high : numerical
        指定范围的上界
    n : int
        生成序列的个数，实际生成的个数会大于`n`

    Returns
    -------
    ndarray
        均匀的序列，范围大于`(low, high)`
    """
    # 在分箱初始化中，可能会使用，生成初始的分箱区间
    def nice_number(x):
        exp = np.trunc(np.log10(abs(x)))
        f = abs(x) / 10 ** exp
        if f < 1.5:
            nf = 1.
        elif f < 3.5:
            nf = 2.
        elif f < 7.5:
            nf = 5.
        else:
            nf = 10.
        return np.sign(x) * nf * 10.0 ** exp
    d = abs(nice_number((high - low) / (n - 1)))
    min_y = np.floor(low / d) * d
    max_y = np.ceil(high / d) * d
    return np.arange(min_y, max_y + 1, d)
