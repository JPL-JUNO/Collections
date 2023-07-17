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


def calculate_chi2(cnt: DataFrame, bin1: int, bin2: int):
    # 计算出四联表
    # 因为最后一列是求和列，舍弃，只需要Aij
    Aij = cnt.iloc[[bin1, bin2], :-1].values
    Ri = Aij.sum(axis=1)
    Cj = Aij.sum(axis=0)
    Ri[Ri == 0] = .1
    Cj[Cj == 0] = .1
    Aij[Aij == 0] = .1
    N = Aij.sum()
    Eij = Ri.reshape(-1, 1) * Cj
    Cj = Aij.sum(axis=0)
    Cj[Cj == 0] = .1

    contingency_table = np.vstack((Aij, Cj))
    Eij = contingency_table[:, -1].reshape(-1, 1) * \
        contingency_table[-1, :].reshape(1, -1) / contingency_table[-1, -1]
    chi2 = ((Aij[:, :-1] - Eij[:-1, :-1]) ** 2 / Eij[:-1, :-1]).sum()
    print(chi2)
    print(Eij)
    print(Eij[:-1, :-1])
    Eij = Aij[:, -1].reshape(-1, 1) * Cj[:-1] / Cj[-1]
    print(Eij)
    print(Aij[:, :-1])
    chi2 = ((Aij[:, :-1] - Eij)**2 / Eij).sum()
    print(chi2)
    return chi2


if __name__ == '__main__':
    df = pd.read_csv('tidy.data', names=['A', 'B', 'C', 'D', 'label'])[:100]
    cnt = dsct_init(df, list('ABCD'))
    calculate_chi2(cnt, 1, 2)
