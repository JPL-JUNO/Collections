"""
@Description: 过滤特征的方法
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-15 14:21:10
"""
from pandas import DataFrame
import pandas as pd
from collections import defaultdict
RISK_MIN = 1.1
RISK_MAX = .9


def filter_by_risk(df: DataFrame, category: list[str], global_mean: float = .5,
                   label: str = 'label',
                   risk_interval: tuple = (RISK_MAX, RISK_MIN)) -> defaultdict[bool]:
    """用于确定分类变量的特征重要性，基于risk方式，参考《Machine Learning BookCamp》3.1.4 feature importance 

    Parameters
    ----------
    df : DataFrame
        用于表示数据的数据框，需包含标签和特征
    category : str
        表示分类变量的字段，用于分组聚合
    global_mean : float, optional
        所有样本的平均水平，用于按照category分组后，判断组内target比例是否变化, by default .5
    label : str, optional
        表示df中的target变量, by default 'label'
    risk_interval : tuple, optional
        risk的上界和下界，必须存在某个分组的risk大于RISK_MIN or less than RISK_MAX，这样才能将该特征保留, by default (RISK_MAX, RISK_MIN)

    Returns
    -------
    defaultdict[bool]
        分类变量是否保存的字典
    """
    # 风险值的区间，如果全部分值落在这个区间，这个分类变量不应该保留，返回False
    keep_flg = defaultdict(bool)
    for col in categorical:
        risk_iv = pd.Interval(left=risk_interval[0], right=risk_interval[1])
        df_group = df.groupby(by=col)[label].agg(['mean'])
        df_group['diff'] = df_group['mean'] - global_mean
        df_group['ratio'] = df_group['mean'] / global_mean
        keep_flg[col] = df_group['ratio'].map(lambda x: x not in risk_iv).any()
    return keep_flg


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

df = pd.DataFrame(data={'A': list('ababbbaa'),
                        'B': [1, 2, 3, 3, 2, 2, 3, 1]})
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['churn'] = (df['churn'] == 'Yes').astype(int)
df = df.dropna(axis=1)


keep_flag = filter_by_risk(df, category=categorical,
                           global_mean=0.2654, label='churn')
