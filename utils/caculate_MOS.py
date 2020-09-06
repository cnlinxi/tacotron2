# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 13:56
# @Author  : MengnanChen
# @FileName: caculate_MOS.py
# @Software: PyCharm

import math
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import t


def calc_mos(data_path: str):
    '''
    计算MOS，数据格式：MxN，M个句子，N个试听人，data_path为MOS得分文件，内容都是数字，为每个试听的得分
    :param data_path:
    :return:
    '''
    data = pd.read_csv(data_path)
    mu = np.mean(data.values)
    var_uw = (data.std(axis=1) ** 2).mean()
    var_su = (data.std(axis=0) ** 2).mean()
    mos_data = np.asarray([x for x in data.values.flatten() if not math.isnan(x)])
    var_swu = mos_data.std() ** 2

    x = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.asarray([var_uw, var_su, var_swu])
    [var_s, var_w, var_u] = solve(x, y)
    M = min(data.count(axis=0))
    N = min(data.count(axis=1))
    var_mu = var_s / M + var_w / N + var_u / (M * N)
    df = min(M, N) - 1  # 可以不减1
    t_interval = t.ppf(0.975, df, loc=0, scale=1)  # t分布的97.5%置信区间临界值
    interval = t_interval * np.sqrt(var_mu)
    print('{} 的MOS95%的置信区间为：{} +—{} '.format(data_path, round(float(mu), 3), round(interval, 3)))


if __name__ == '__main__':
    data_path = ''
    calc_mos(data_path)
