"""
@Description:
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-08 15:24:09
"""
import timeit
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from method.frame.util import check_y, x_variable
from method.frame.info_value import information_value


def var_filter(dt: DataFrame, y: str, x=None,
               iv_limit: float = .02, missing_limit: float = .95, identical_limit: float = .95,
               var_rm: list | None = None, var_kp: list | None = None,
               return_rm_reason: bool = False, positive: str = 'bad|1') -> dict:
    """根据一些准则过滤变量

    Args:
        dt (DataFrame): 数据，包含features和target
        y (str): target
        x (_type_, optional): features waited to filter, if not specified all features in dt will be used. Defaults to None.
        iv_limit (float, optional): 被保留特征最小的information value. Defaults to .02.
        missing_limit (float, optional): 被保留特征最大的确实比例，超过则被删除. Defaults to .95.
        identical_limit (float, optional): 被保留特征某一单一值最大占比，超过则被删除. Defaults to .95.
        var_rm (list | None, optional): 手动传入的必被删除变量. Defaults to None.
        var_kp (list | None, optional): 强制保留的变量，这些变量不需要满足上述条件. Defaults to None.
        return_rm_reason (bool, optional): 是否返回被删除原因. Defaults to False.
        positive (str, optional): 正类的标识符. Defaults to 'bad|1'.

    Returns:
        DataFrame: _description_
    """
    start_time = timeit.default_timer()
    df = dt.copy(deep=True)
    if x is not None:
        x = [x] if isinstance(x, str) else x
        df = df[[y] + x]
    # 检验输入数据的合理性，（仅适用于二分类样本）
    df = check_y(df, y, positive)

    x = x_variable(df, y, x)

    if var_rm is not None:
        if isinstance(var_rm, str):  # 如果传入的强制删除变量为字符串，则转化为列表
            var_rm = [var_rm]
        x = list(set(x).difference(set(var_rm)))  # 将被删除的变量（字段）中特征列表 x 中移除
    if var_kp is not None:
        if isinstance(var_kp, str):
            var_kp = [var_kp]
        var_kp2 = list(set(x).intersection(set(var_kp)))
        if set(var_kp).difference(set(var_kp2)):
            warnings.warn('存在{0:4}无效的保留字段，因为数据中不存在：\n'.format(
                len(set(var_kp).difference(set(var_kp2))), list(set(var_kp).difference(set(var_kp2)))))
    iv_ser = information_value(df, y, x, order=False)

    # def nan_rate(a): return a[a.isnull()].size / a.size
    # na_pct = df[x].apply(nan_rate).reset_index(
    #     name='missing_rate').rename(columns={'index': 'variable'})

    # 各字段的缺失率，columns作为index
    na_pct: Series = df[x].isnull().sum() / df.shape[1]
    identical_pct: Series = df[x].apply(
        lambda col: col.value_counts(normalize=True).max())
    df_indicator = pd.concat([na_pct, identical_pct, iv_ser], axis=1)
    mask = (df_indicator[0] < missing_limit) & (
        df_indicator[1] < identical_limit) & (df_indicator[2] > iv_limit)

    x_selected = df_indicator[mask].index.tolist()
    if var_kp is not None:
        x_selected = list(set(x_selected + var_kp))
    df_selected = df[x_selected + y]
    running_time = timeit.default_timer() - start_time
    if (running_time > 10):
        print(
            f'特征过滤已完成，耗时：{running_time:.4f}s\n{df.shape[1]-len(x_selected):3}个变量被移除')
        print('Variable filtering on {} rows and {} columns'.format(
            dt.shape[0], dt.shape[1]))

    if return_rm_reason:
        x_threshold = df_indicator.assign(
            info_value=lambda x: [
                f'info value <{iv_limit}' if i else np.nan for i in (x[2] < iv_limit)],
            missing_rate=lambda x: [
                f'missing rate <{missing_limit}' if i else np.nan for i in (x[1] > missing_limit)],
            identical_rate=lambda x: [
                f'identical rate <{identical_limit}' if i else np.nan for i in (x[0] < identical_limit)],
        )
        if var_rm is not None:
            x_remove_reason = pd.concat(
                [x_threshold, pd.Series(data=['force remove'] * len(var_rm), index=var_rm)], axis=1)
        if var_kp is not None:
            x_remove_reason = pd.concat(
                [x_remove_reason, pd.Series(data=[np.nan] * len(var_kp), index=var_kp)], axis=1)
        x_remove_reason = x_threshold.dropna(how='all')
        return {'data': df_selected, 'rm': x_remove_reason}
    else:
        return {'data': df_selected}


if __name__ == '__main__':
    pass
