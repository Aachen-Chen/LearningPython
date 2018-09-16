import copy
import warnings
from time import strftime
import datetime
from datetime import timedelta
from math import sqrt

import numpy as np
import pandas as pd
from pandas import Series
import cvxopt
from cvxopt import solvers
import matplotlib.pyplot as plt

from b_model.port_raroc import get_ports_rtn_series, get_index_rtn_series
from b_model.common import Period, Index
# from b_model.barra_multiple_factor.barra_factor import *
from P_FirstSeafront.barra_factor import *

def imports():
    return

"""
First developed by Aachen, QHKY

优化框架：
    （其中x为优化变量）
    线性规划 lp:
        minimize    P.T x
        subject to  Gx <= h
                    Ax = b
    二次规划 qp:
        minimize    (1/2)x.T P x + q.T x
        s.t.        Gx <= h
                    Ax = b
    二阶锥规划 SOCP:
        minimize    f.T x
        s.t.        ||Ax + b||2 <= c.T x
                    Fx = g

惯例：
    个股i，总数n
    因子j，总数k
    日期t，总数T

优化类型：
    最小化预期风险：
        qp:
            P = (XF[t-1]X[t-1] + sig[t-1] )
    最大化绝对收益：
        lp:
            P = -R'                                         以预期收益率  为优化目标
            R'= X[t-1]F[t-1] + spec[t-1].mean()             以预测的因子收益、特定收益  为优化目标
            R'= R' - f(w[t-1])                              考虑调仓
    最小化跟踪误差：
        qp:
            P = (XF[t-1]X[t-1] + sig[t-1])                  (w_port - w_bench)**2 = w_port ** 2 - 2 w_p * w_b + 常数项
            Q = -(XF[t-1]X[t-1] + sig[t-1]) * w_bench       - 2 w_p * w_b 交互项
    最大化风险调整后收益：
        qp:
            P = 2λ(XFX + sig)                              λ:[0, +∞)，[风险中性, 风险厌恶)。系数2是因为cvxopt目标函数的二次项默认乘以1/2
            Q = -R'
    最大化夏普比率：
        qp（不确定）：
            P = -R' / (XFX + sig)
约束条件：
    禁止做空，没有资金
        s.t.
            A = [1],        b = [1]                         权重和为0
            G = [1],        h = [0]                         禁止卖空
    允许做空：
        s.t.
            G = None,       h = None        允许
    风格/行业中性：
        s.t.
            A = [1, MX],   b = [1, 0]       相应暴露为0。若只考虑风格中性，X[t]只包含风格因子以加快速度。
    VaR约束：
        二阶锥规划（SOCP）
    预期风险低于某范围：
        可以转化为qp

优化函数 -> t日权重：
    Parameters, 参数检查：
        F       k * k
        X       n * k
        sig     n
        spec    n
        R'      n
        A, b, G, h
    Return:
        w       1 * n
"""

@log
def neutral_factor(*args: Factor) -> list:
    factor = []
    for fac in args:
        factor.append(fac)
    return factor

class Neutral_Port_Opt(object):
    def __init__(self, t_date: str, benchmark: Index = Index.i000300, factorgroup: FactorGroup = FactorGroup.Indices,
                 stock_list: list = None, max_weight: float = 0.03):
        self.t_date = t_date
        self.bench = benchmark
        self.factorgroup = factorgroup
        self.max_weight = max_weight
        self.neu_fac = []
        self.non_bench_stock = []
        return

    def set_neutral_fac(self, *facs: Factor) -> list:
        for fac in facs:
            self.neu_fac.append(fac)
        return

    def set_stock_exp_rtn(self, exp_rtn: DataFrame):
        self.selected_stock_exp_rtn = exp_rtn
        self.bench_list = get_bench_weight(self.t_date, self.t_date, self.bench).reset_index()['ASSETID'].drop_duplicates().tolist()
        for stock in self.selected_stock_exp_rtn.iloc[:, 1]:
            if stock not in self.bench_list:
                self.non_bench_stock.append(stock)
        return

    def get_min_var_port(self) -> Series:
        self.expo = get_barra_expo(self.t_date, self.t_date, self.bench, self.factorgroup, self.non_bench_stock)
        self.fac_cov = get_barra_cov(self.t_date, self.t_date, self.factorgroup)
        self.wgt = min_var_neutral_portfolio(self.fac_cov, self.expo, self.neu_fac, self.max_weight)
        return self.wgt

    def get_max_rtn_port(self, exp_month: int) -> Series:
        self.expo = get_barra_expo(self.t_date, self.t_date, self.bench, self.factorgroup, self.non_bench_stock)
        self.t_date_end = (pd.to_datetime(self.t_date) + timedelta(days = exp_month*30)).strftime("%Y-%m-%d")
        self.exp_rtn = get_selected_stock_rtn(self.t_date, self.t_date_end, Period.DAY, self.non_bench_stock, self.bench
            ).pivot_table(values='RTN', index='T_DATE', columns="ASSETID").mean()
        for i, row in self.selected_stock_exp_rtn.iterrows():
            self.exp_rtn.loc[row["T_DATE"]] = row["exp_rtn"]
        self.wgt = max_rtn_neutral_portfolio(self.exp_rtn, self.expo, self.neu_fac, self.max_weight)
        return self.wgt

    def get_min_te_port(self) -> Series:
        self.expo = get_barra_expo(self.t_date, self.t_date, self.bench, self.factorgroup, self.non_bench_stock)
        self.fac_cov = get_barra_cov(self.t_date, self.t_date, self.factorgroup)
        self.bench_wgt = get_bench_weight(self.t_date, self.t_date, self.bench).loc[self.t_date]
        self.wgt = min_tracking_error_neutral_portfolio(self.fac_cov, self.expo, self.bench_wgt, self.neu_fac, self.max_weight)
        # 优化函数可以对齐矩阵，只需要确保传入的矩阵没有双重索引，且<<矩阵方向>>正确（bench_wgt必须是行向量）。
        return self.wgt

    def port_statistics(self):

        return

@log
def neutral_portfolio(cov_mat: DataFrame, expo: DataFrame, neutral_list: list, max_weight: float = 0.03) -> Series:
    n, k = expo.shape

    # min(Var) = min(w.T * (X * Cov(F) * X.T) * w)
    P = cvxopt.matrix(np.dot(expo, np.dot(cov_mat, expo.T)).astype(np.float64))    # n * n
    q = cvxopt.matrix(0.0, (n, 1))
    print("Neutral include:", neutral_list, "P size:", P.size)

    # weight >= 0
    G = -np.identity(n)
    h = np.zeros((n, 1))

    # weight <= max_weight
    G = np.vstack((G,
                   np.identity(n)))
    h = np.vstack((h,
                   np.full((n, 1), max_weight)))

    # port_expo_j = np.dot( weight, expo_fac_j ) = 0
    for neu_fac in neutral_list:
        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            np.zeros((2, 1))    ))

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # sum(w) = 1
    A = cvxopt.matrix(1.0, (1, n))
    b = cvxopt.matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    port_wgt = pd.Series(sol['x'], index = expo.index)

    print("------ Opt solved. ------")
    return port_wgt


@log
def min_var_neutral_portfolio(cov_mat: DataFrame, expo: DataFrame, neutral_list: list, max_weight: float = 0.03) -> Series:
    n, k = expo.shape

    P = cvxopt.matrix(np.dot(expo, np.dot(cov_mat, expo.T)).astype(np.float64))    # n * n
    q = cvxopt.matrix(0.0, (n, 1))
    print("Neutral include:", neutral_list, "P size:", P.size)

    G = -np.identity(n)
    h = np.zeros((n, 1))

    G = np.vstack((G,
                   np.identity(n)))
    h = np.vstack((h,
                   np.full((n, 1), max_weight)))

    for neu_fac in neutral_list:
        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            np.full((2, 1), 0.005)    ))
        # print("expo:", expo[neu_fac].values)

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    A = cvxopt.matrix(1.0, (1, n))
    b = cvxopt.matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    port_wgt = pd.Series(sol['x'], index = expo.index)

    print("------ Opt solved. ------")
    return port_wgt

@log
def max_rtn_neutral_portfolio(exp_rtn: Series, expo: DataFrame, neutral_list: list, max_weight: float = 0.03) -> Series:
    n, k = expo.shape

    c = cvxopt.matrix(exp_rtn.values)
    print("Neutral include:", neutral_list, "c size:", c.size)

    G = -np.identity(n)
    h = np.zeros((n, 1))

    G = np.vstack((G,
                   np.identity(n)))
    h = np.vstack((h,
                   np.full((n, 1), max_weight)))

    for neu_fac in neutral_list:
        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            np.zeros((2, 1))    ))

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    A = cvxopt.matrix(1.0, (1, n))
    b = cvxopt.matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    port_wgt = pd.Series(sol['x'], index = expo.index)

    print("------ Opt solved. ------")
    return port_wgt


@log
def min_tracking_error_neutral_portfolio(cov_mat: DataFrame, expo: DataFrame, bench_wgt: Series, neutral_list: list, max_weight: float = 0.03) -> Series:
    n, k = expo.shape

    P = cvxopt.matrix(np.dot(expo, np.dot(cov_mat, expo.T)).astype(np.float64))    # n * n
    q = cvxopt.matrix(-np.dot(P, bench_wgt.T))
    print("Neutral include:", neutral_list, "P size:", P.size)

    G = -np.identity(n)
    h = np.zeros((n, 1))

    G = np.vstack((G,
                   np.identity(n)))
    h = np.vstack((h,
                   np.full((n, 1), max_weight)))

    for neu_fac in neutral_list:
        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            np.zeros((2, 1))    ))

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)


    A = cvxopt.matrix(1.0, (1, n))
    b = cvxopt.matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    port_wgt = pd.Series(sol['x'], index = expo.index)

    print("------ Opt solved. ------")
    return port_wgt


"""
一次优化
"""


"""
回测
数据字典：（请注意时间范围）
labels：
    date_list
    date_list_shift_1
    fac_list            风格/行业因子列表
    stock_list          选股范围                               基准股票范围变更时，直接纳入新权重股
    neu_fac_list        需要中性的因子

t：
    fac_rtn             因子回报            f    k * t                              barra_factor.get_barra_fact_rtn()
    fac_cov(sig)        因子协方差矩阵      F    k * k * t     两个index            barra_factor.get_barra_cov()
    expo([t, end+1])    因子暴露矩阵        X    n * k * t     两个index            barra_factor.get_barra_expo()
    stock_spec_rtn      特殊回报            spec n * t                              barra_factor.get_barra_spec_rtn()
    stock_spec_sig      特殊风险矩阵        sig  n * t                              barra_factor.get_barra_spec_risk()
t+1：
    stock_rtn           个股实际收益率      R    n * t
    stock_rtn_exp       个股预期收益率      R'   n * t'        t' 样本外区间长度

t+1：
    port_rtn
    bench_rtn           基准收益            Rb   t                                  get_benchmark_rtn_series()
    port_wgt
    bench_wgt(可忽略)   基准权重            wb   n * t                              AIndexHS300Weight.i_weight
    port_expo
    bench_expo          基准暴露                                                    SQL

工作方法调整：
    先想再做。
    没有大的思路，各个小功能测试的先后，参数的轻重。
        为了实现一些小功能花了太多时间
        老在代码层面修改，而不是去找API
    数据流的格式不一致，名称记不住。学习一下？
        尽量在同一个DF上操作。
    没有进度控制。没有随时跟踪耗时点。

推荐实践：
基础代码：
    生成新DF：用字典
处理时间变化：（涉及所有）
    日期处理：shift_1：
        原始序列：date_list = get_fac_cov(start, end).index
        原始序列shift_1：date list = get_stock_rtn(start+1, end+1).index
            不应直接用date_list右移。因为不是连续的日期列表。
            start + 1 应该用 timedelta 实现。
        下一日：t_date_shift_1 = date_list_shift_1[date_list.index(t_date)]
            不应该用 timedelta 右移。因为不是连续日期。
            注意，shift_1和原date_list，同一个index，对应前后两天。
            直接用date_list的下一个index，可能出现out of bound。
数据对齐：
    矩阵乘法一律直接用 DataFrame / Series，以利用其自动匹配相应行列的功能。
        （除了一维向量，np.dot(DF, DF.T) 矩阵运算都要记得对齐行列长度）
计算
    连乘：rtn转化为cum_rtn时，必须先x.fillna(0) 再加一！

开发和测试：
基础开发：
    数据部分 - 扩展性（注意问题）：
        处理时间变化：（涉及所有）
            节假日处理：时移一位，应根据时移前的工作日后移，不能简单用自选日期后移一位.
                不能使用(pd.to_datetime(end) + timedelta(days = 1)).strftime("%Y-%m-%d")的公式。
                应该通过date_list[-1], date_list[index + 1]的方式来获取下一有效日期。
        处理样本变化：
            在进入优化函数前，只包含当日有数据的股票，并对齐。
            权重股数量变化：（涉及expo, rtn, wgt）
                expo表：获取整个时间范围覆盖的所有股票。如无数据，则fillna(0)。
                wgt / port_rtn 表：沿用此做法，并确保数据缺失的股票无权重、收入。
                rtn表：只包含当日有数的股票。作为对齐的尺度。
                变化当日处理：见循环优化部分
            因子数量：暂不处理
        可选因子组合、基准、股票
            基准：默认沪深300；因子：默认风格因子。如需返回所有股票/所有因子，需传入实参 None。(但不能选取两类因子或两个基准）
                rtn表：统一用 V_INDEX_RTN_DETAIL，实现可选bench
                    →其他bench的选取
            选股：默认None。
            统一用默认形参、枚举类型。
            加入自选股：（涉及expo, rtn, wgt）
                统一加入union字段。
        检查一遍函数使用。更新形参的调整。
        检查统一stock/fac顺序。
        加入中性因子。
        弄个类来统一所需数据。
    结果部分
        能否优化，组合收益、回撤、夏普比率相比hs300如何；
        能否实现中性，组合/基准的因子暴露；
        测度调仓成本
优化器部分：
    min_var: 只包括风格因子，只测试沪深300权重股，不包括特殊回报
        能否包含调仓成本
    max_rtn: 包括简单的收益预测
        包括个股特殊回报
            NB用于特殊回报预测
        因子收益预测对优化的影响
    min_tracking_error
        能否实现降低tracking risk
    限定个股数量
    各个因子的收益：barra_dlyfacret
    面向对象实现
系统上线及逻辑接口：

补充：
    代码注释

问题：
    因子收益如何预测？对优化影响大吗？
"""

""" 参数设置 """
# start, end_shift1 = '2017-06-05', '2017-06-08'
start, end_shift1 = '2016-11-07', '2017-06-12'
# start, end_shift1 = '2016-12-21', '2017-01-01'
# start = '2017-06-07'
# end = (pd.to_datetime(start) + timedelta(days = 4)).strftime("%Y-%m-%d")

benchmark = Index.i000300
fac_group = FactorGroup.Indices
neu_fac_list = neutral_factor(Factor.SIZENL)     # Factor.BETA, Factor.SIZE
turnover_period = Period.MONTH

""" 数据读取、对齐 """
""" 别忘了这些都是双重索引！（第二层索引的level=1) 万恶之源 """
fac_cov = get_barra_cov(start, end_shift1, fac_group)

stock_rtn = get_selected_stock_rtn(start, end_shift1, Period.DAY, [], Index.i000300)
""" 2018.04.19 回测部分：正在考虑将调仓日设在每月初，通过stock_rtn表确定月初日期，再调取expo，以提高运行速度。 """
""" 2018.04.19 回测部分：暂停回测部分工作。先设计最小化跟踪误差，最大化信息比率的优化器。 """
stock_list_total = stock_rtn.reset_index()['ASSETID'].drop_duplicates().reset_index(drop=True).tolist()
print("Total number of stock:", len(stock_list_total))

d_list_barra = fac_cov.reset_index()['T_DATE'].drop_duplicates().reset_index(drop=True).tolist()
d_list_stock = stock_rtn.reset_index()['T_DATE'].drop_duplicates().reset_index(drop=True).tolist()
print("Length: d_list_barra, d_list_stock:", len(d_list_barra), len(d_list_stock))
print("d_list_barra", d_list_barra)
print("d_list_stock", d_list_stock)

""" 日期检查 """
d_list = pd.Index(d_list_barra).intersection(pd.Index(d_list_stock)).sort_values().tolist()
""" 如果日后加入历史平均收益作为预测收益的功能，收益频率选月时，需要将开始日调到最接近的月初，结束日调到最接近的月末。 """

"""
当前任务：
    把特异风险加入P
    完成一个评估模块
    完成剩下两个的unittest
    完成注释和整理
"""

d_list_shift1 = d_list[1:]
d_list.pop(-1)
start, end, start_shift1, end_shift1 = d_list[0], d_list[-1], d_list_shift1[0], d_list_shift1[-1]
print("date_list", d_list)
print("date_list_shift", d_list_shift1)
print("Total number of dates:", len(d_list) + 1)

""" 稍后将这一块移到 stock_rtn 代码段之前 """
expo = get_barra_expo(start, end_shift1, Index.i000300, fac_group)
expo.sort_index(level=1, ascending=True, inplace=True)
print("Expo shape:", expo.shape)

fac_list = fac_cov.loc[start].index.tolist()
n, k, T = len(stock_list_total), len(fac_list), len(d_list)
# idx = pd.IndexSlice

""" 检查无效列，检查是否对齐 """
""" !!!要检查各期个股序列一致!!! """

""" 初始化组合收益/权重/暴露，注意：rtn、wgt 和 port_expo 都是t+1的。 """
port_rtn = pd.Series(np.zeros(T), index=d_list_shift1)
port_wgt = pd.DataFrame(np.zeros((T, n)), index = d_list_shift1, columns=stock_list_total)
port_expo = pd.DataFrame(np.zeros((T, k)), index=d_list_shift1, columns=fac_list)

bench_rtn = get_index_rtn_series('000300.SH', start_shift1, end_shift1, None).set_index('T_DATE')['IDX_RTN_RATE']
bench_wgt = get_bench_weight(start_shift1, end_shift1, benchmark)
bench_expo = pd.DataFrame(np.zeros((T, k)), index = d_list_shift1, columns=fac_list)

for index, t0 in enumerate(d_list):
    """
    滚动导入当日因子暴露、协方差矩阵。求出的wgt，与t日r相乘，记入t日的组合收益
    选股范围变化当日的处理：
        t0日没有，t1有：t+1日不投资，t+2才开始投。 wgt.loc[t_d_shift1, t_stock_list_shift1]，第二个参数规定了t+1日有数的，才会有权重。t+1日没有数的，即便优化器算出权重，也不放入，默认为0.（默认即wgt矩阵一开始的赋值，np.zeros)
        t0日有数，t1没：不投资。实现方法相同。
    日期缺失值处理：
        直接取交集。则循环阶段必须用 字符串日期 作索引。
            注意：日期交集只用于port数据的index。bench数据从外部读取，可能包括port缺失的数据。
            合并port 和 bench df时，会因为index不同，造成port的缺失日显示NaN。务必 .fillna(0)
        缺少expo/cov：当日 rtn 照常计算，当日 expo 维持昨日，次日wgt 维持今日  （未开发）
        缺少stock_rtn：当日 rtn 为0，当日 expo 照常计算，wgt 照常计算         （未开发）
    """

    print("--------- Solving: ", t0, " ---------")
    """ t日的expo向量，只包含当天有数的股票 """
    print(stock_rtn.reset_index()['T_DATE'].drop_duplicates().reset_index(drop=True).tolist()[0:2])
    t0_stock_list = stock_rtn.loc[t0].index.tolist()
    t0_fac_cov, t0_expo = fac_cov.loc[t0], expo.loc[t0].loc[t0_stock_list, :]

    """ 次日的expo和rtn，用以计算port_rtn / expo """
    t1 = d_list_shift1[index]
    t1_stock_list = stock_rtn.loc[t1].index.tolist()
    t1_stock_rtn, t1_expo = stock_rtn.loc[t1], expo.loc[t1].loc[t1_stock_list, :]
    # print(t_stock_rtn)
    # s_exp_rtn = pd.Series(np.dot(expo, fac_exp_rtn.T), index=securities_list)
    # s_tar_rtn = s_exp_rtn.quantile(0.7)

    print("Shape of cov, expo:\n", t0_fac_cov.shape, "\n", t0_expo.shape)
    port_wgt.loc[t1, t1_stock_list] = min_var_neutral_portfolio(t0_fac_cov, t0_expo, neu_fac_list)    # 式A
    port_wgt.fillna(0.0, inplace=True)
    """ 式A
    向df传Series，若用loc规定了具体列，则：
        指定列以外的不变（t0做了优化、t1没有数的，权重仍为0）
        指定列内，没有传入的，变成NaN。需要后续fillna（t0没有优化，但t1加入选股范围的，不投资，权重为0）
    """
    port_rtn.loc[t1] = np.dot(port_wgt.loc[t1, t1_stock_list], t1_stock_rtn)
    port_expo.loc[t1, :] = np.dot(port_wgt.loc[t1, t1_stock_list], t1_expo)
    bench_expo.loc[t1, :] = np.dot(bench_wgt.loc[t1, t1_stock_list], t1_expo)
    """ 无需fillna，因为port_rtn / port_expo 都不包含个股数据。"""

""" Summary: rtn, wgt, port/bench expo """
rtn = pd.DataFrame({'port_rtn': port_rtn, 'bench_rtn': bench_rtn}).fillna(0)
# 合并index分别由外部决定和由内部日期取交集的两个df，必须fillna。
print("rtn", rtn)
cum_rtn = pd.DataFrame(rtn.apply(lambda x: (x.fillna(0)+1).cumprod(), axis=0), index=rtn.index)
print("cum_rtn", cum_rtn)
# 2018.03.27: 牛皮！显著领先！

""" Visualization """
# 绘图准备
fac_rtn = get_barra_fac_rtn(start, end, fac_group)
fac_cum_rtn = pd.DataFrame(fac_rtn.apply(lambda x: (x.fillna(0)+1).cumprod(), axis = 0), index = fac_rtn.index)
neutral_factor_expo = pd.DataFrame(
    {"bench": bench_expo.loc[:, Factor.SIZENL],
     "port": port_expo.loc[:, Factor.SIZENL]    }   )

# port_wgt['sum'] = port_wgt.apply(lambda x: x.sum(), axis=1)
# fac_rtn_rolling = fac_rtn.apply(lambda x: x.rolling(window=20, center=False).mean())

# 绘图区
# rtn.plot(grid=True).axhline(y=1, color="black", lw=2)
# fac_rtn_rolling.plot(grid=True)
# factor_price.plot(grid=True)
# bench_expo.plot(grid=True).axhline(y=1, color="black", lw=2)
port_expo.plot(grid=True).axhline(y=1, color="black", lw=2)
neutral_factor_expo.plot(grid=True).axhline(y=1, color="black", lw=2)
cum_rtn.plot(grid=True).axhline(y=1, color="black", lw=2)
fac_cum_rtn.plot(grid=True).axhline(y=1, color="black", lw=2)

# fac_rtn.plot(grid=True).axhline(y=1, color="black", lw=2)
# port_wgt.plot(grid=True).axhline(y=1, color="black", lw=2)
plt.show()

print("Exit: test_min_var_portfolio()")


def port_analysis(port_rtn: DataFrame, bench_rtn: DataFrame, period: Period = None, if_annualized: bool = None):
    """ """
    """
    beta, alpha, volatility, downside_deviation, tracking_error, max_drawdon
    sharpe_ratio, information_ratio, calmar_ratio, sortino_ratio
    """
    """ beta, alpha """
    # rtn_series = pd.merge(port_rtn_series, bench_rtn_series, how='inner', right_index=True, left_index=True)
    rtn = pd.DataFrame({'port_rtn': port_rtn, 'bench_rtn': bench_rtn}).fillna(0)
    rtn_port, rtn_bench = np.array(rtn['port_rtn']), np.array(rtn['bench_rtn'])
    independent_matrix = np.vstack([rtn_bench, np.ones(len(rtn_bench))]).T
    beta, alpha = np.linalg.lstsq(independent_matrix, rtn_port)[0]

    if period == 'WEEK':
        alpha = (52 * alpha if if_annualized else alpha)
    elif period == 'MONTH':
        alpha = (12 * alpha if if_annualized else alpha)
    else:
        alpha = (250 * alpha if if_annualized else alpha)

    """ volatility, downside deviation, tracking error """
    if if_annualized:
        if period == 'WEEK':
            volatility = sqrt(52) * np.nanstd(port_rtn, ddof=1)
            downside_deviation = sqrt(52) * np.nanstd(port_rtn.loc[port_rtn < 0])
        elif period == 'MONTH':
            volatility = sqrt(12) * np.nanstd(port_rtn, ddof=1)
            downside_deviation = sqrt(52) * np.nanstd(port_rtn.loc[port_rtn < 0])
        else:
            volatility = sqrt(250) * np.nanstd(port_rtn, ddof=1)
            downside_deviation = sqrt(52) * np.nanstd(port_rtn.loc[port_rtn < 0])
    else:
        np.nanstd(port_rtn, ddof=1)
        downside_deviation = np.nanstd(port_rtn.loc[port_rtn < 0])

    tracking_error = np.nanstd(port_rtn - bench_rtn)


    """  """

    """  """

    """  """

    return


"""
滚动计算：
    port_rtn = []
    对于  样本外的每一天：
        w[t] = 优化函数（参数[t-1]）
        port_rtn.append( np.dot(R[t].T, w[t]) )

基准比较：
    bench_rtn = Rb
    bench_rtn.iloc[o:in_num] = 0
    rtn = DataFrame({ 'port_rtn': port_rtn,
                      'bench_rtn': bench_rtn })
    cum_rtn = DataFrame(rtn.apply(lambda x: pd.Series(np.cumprod(np.nan_to_num(np.array(x))+1))))
    cum_rtn.index = bench_rtn['T_Date']

    cum_rtn.plot(grid = True)
    plt.show()
"""

# print(wgt)

# if __name__ == '__main__':
# """ 检验并删去 """
# # cov_mat.drop(['COUNTRY'], axis = 1, inplace=True)
# # cov_mat.drop(['COUNTRY'], axis = 0, level= 1, inplace=True)
# # expo.drop(['COUNTRY'], axis = 1, inplace=True)
#
# """ 检查是否整齐 """
# print("Expo columns:", list(zip(expo.columns.tolist(), cov_mat.columns.tolist(), cov_mat.index.tolist())))
# # print("Shape", cov_mat.shape, expo.shape)
# # print("Cov rank:", np.linalg.matrix_rank(cov_mat))


@log
def test():
    result = get_ports_rtn_series(['000300.SH'], '2017-06-05', '2017-06-16', Period.WEEK)
    print(result)
    print(len(result.index))
    # for i in range(result.index)

    return 0


