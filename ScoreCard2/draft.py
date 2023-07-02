"""
@Description: 函数草稿
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-01 20:03:36
"""
from time import ctime
import sys
sys.path.append('./')
sys.path.append('../')


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
    pass


if __name__ == '__main__':
    data = data_loading('../ScoreCard2/tidy.data')
