"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-08 15:24:09
"""
import timeit
from pandas import DataFrame
from method.frame.util import check_y


def var_filter(dt: DataFrame, y: str, x=None,
               iv_limit: float = .02, missing_limit: float = .95, identical_limit: float = .95,
               var_rm: list | None = None, var_kp: list | None = None,
               return_rm_reason: bool = False, positive: str = 'bad|1') -> DataFrame:
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

    pass


if __name__ == '__main__':
    pass
