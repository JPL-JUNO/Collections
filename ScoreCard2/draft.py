"""
@Description: 函数草稿
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-01 20:03:36
"""
from time import ctime
import numpy as np
from pandas import DataFrame
import sys
sys.path.append('./')
sys.path.append('../')
import pandas as pd


def data_loading(file: str) -> list[list]:
    instances = []
    with open(file, 'r') as fp:
        for line in fp:
            line = line.strip('\n')
            if line != '':
                instances.append(line.split(','))
    return instances


def split(instances, i):
    """获取第i个特征和target

    Parameters
    ----------
    instances : _type_
        _description_
    i : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    log = []
    for line in instances:
        log.append([line[i], line[4]])
    return log


def count(log):

    log_cnt = []
    # 按照特征进行升序排序
    log.sort(key=lambda log: log[0])
    i = 0
    while (i < len(log)):
        cnt = log.count(log[i])
        record = log[i][:]
        record.append(cnt)
        log_cnt.append(record)
        i += cnt
    return log_cnt
    # df[['feature', 'target']].groupby(['feature', 'target']).size().unstack()


def build(log_cnt) -> list[tuple]:
    log_dict = {}
    for record in log_cnt:

        # if record[0] not in log_dict.keys():
        #     log_dict[record[0]] = [0, 0, 0]
        if record[1] == 'Iris-setosa':
            # log_dict[record[0]][0] = record[2]
            log_dict.setdefault(record[0], [0, 0, 0])[0] = record[2]
        elif record[1] == 'Iris-versicolor':
            # log_dict[record[0]][1] = record[2]
            log_dict.setdefault(record[0], [0, 0, 0])[0] = record[2]
        elif record[1] == 'Iris-virginica':
            # log_dict[record[0]][2] = record[2]
            log_dict.setdefault(record[0], [0, 0, 0])[0] = record[2]
        else:
            raise TypeError('Data Exception')
    log_tuple = sorted(log_dict.items())
    return log_tuple


def collect(instances, i):
    log = split(instances, i)
    log_cnt = count(log)
    log_list = build(log_cnt)
    return log_list


def combine(a, b):
    return (a[0], np.array(a[1]) + np.array(b[1]))


# if __name__ == '__main__':
#     data = data_loading('../ScoreCard2/tidy.data')
#     print(combine(('4.4', [3, 1, 0]), ('4.5', [1, 0, 1])))


def chi2(A):
    m = len(A)
    k = len(A[0])
    R = []
    for i in range(m):
        sum = 0
        for j in range(k):
            sum += A[i][j]


def dsct_init(data, feature_cols, target: str = 'label'):

    numerical_cols = data.select_dtypes(include=['float', 'int']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = numerical_cols.drop(
        target) if target in numerical_cols else numerical_cols
    categorical_cols = categorical_cols.drop(
        target) if target in categorical_cols else categorical_cols
    data_type = data[feature_cols].dtypes.value_counts()
    if True:
        # 如果是连续变量
        cnt = pd.crosstab(data[feature_cols[1]],
                          data[target]).sort_index(ascending=True)
        cnt['total'] = cnt.sum(axis=1)
    else:
        pass
    return cnt


def calculate_chi2(cnt: DataFrame, bin1: int, bin2: int) -> float:
    """计算两个组的卡方值

    Parameters
    ----------
    cnt : DataFrame
        已经按照不同取值统计target的数据框
    bin1 : int
        一组的索引值
    bin2 : int
        另一组的索引值，一般是`bin1 + 1`

    Returns
    -------
    float
        两组的卡方值
    """
    # 计算出四联表
    # 因为最后一列是求和列，舍弃，只需要Aij
    Aij = cnt.iloc[[bin1, bin2], :-1].values
    Ri = Aij.sum(axis=1).astype(float)
    Cj = Aij.sum(axis=0).astype(float)
    Ri[Ri == 0] = .1
    Cj[Cj == 0] = .1
    Aij[Aij == 0] = .1
    N = Aij.sum()
    Eij = Ri.reshape(-1, 1) * Cj / N
    chi2 = ((Aij - Eij) ** 2 / Eij).sum()
    return chi2


def merge_adjacent_interval(chi2_list: list, cnt: DataFrame) -> DataFrame:

    min_chi2 = min(chi2_list)
    # 找到最小最的索引所在位置，如果有多个直接一次性合并，而不是每次只找到最靠前的索引(弃用，因为如果存在连续的idx不好处理)
    min_idx = chi2_list.index(min(chi2_list))
    # min_idx = [idx for idx in chi2_list if idx == min_chi2]
    # for idx in min_idx:
    #     cnt.iloc[idx] = cnt.iloc[idx] + cnt.iloc[idx + 1]
    print(min_idx)
    cnt.iloc[min_idx] = cnt.iloc[min_idx] + cnt.iloc[min_idx + 1]
    cnt = cnt.drop(index=cnt.index[min_idx + 1])  # 删除被合并的行（组）
    return cnt


if __name__ == '__main__':
    df = pd.read_csv('tidy.data', names=['A', 'B', 'C', 'D', 'label'])[:100]
    cnt = dsct_init(df, list('ABCD'))
    chi2_list = [calculate_chi2(cnt, idx, idx + 1)
                 for idx in range(len(cnt) - 1)]
    cnt = merge_adjacent_interval(chi2_list, cnt)
