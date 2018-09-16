"""port_opt 提供三种优化器接口

此模块提供了可扩展的组合优化的框架，在常规优化
器的基础上，提供了风格因子中性的条件。实现了3种
优化类型：最小化组合方差、最大化组合预期收益、
最小化跟踪误差。同时包括了一个评价功能。

修改历史：
    陈开琛( kaichenc@andrew.cum.edu ) 2018-03~04 初稿
"""

"""
一、使用方法
    import port_opt.Neutral_Port_Opt
    opt = Neutral_Port_Opt(t_date, bench)   # 实例化优化器并设定条件
    opt.set_stock_exp_rtn(stock_selected)   # 设定*自选股
    opt.set_neutral_fac(neu_fac)            # 设定需要中性的因子
    wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)
    # 求解最优权重。

    * 自选股：优化器默认的universe 为指数
    （hs300等）。如有自选组合，需传入一个
    DataFrame( column = ["ASSETID", "exp_rtn"]）
    其中期望收益为年化。

现有功能限制：（涉及函数）
    只能选取风格因子。（get_barra_expo）
    必须选取一个基准。（get_barra_expo，get_selected_stock_rtn)
    无回测模块。


二、数学关系
    凸优化：
        （其中x为优化变量）
        线性规划 lp:
            minimize    P.T x
            subject to  Gx <= h
                        Ax = b
        二次规划 qp:
            minimize    (1/2)x.T (XF[t-1]X[t-1] + sig[t-1] ) x
            s.t.        Gx <= h
                        Ax = b
        二阶锥规划 SOCP:
            minimize    f.T x
            s.t.        ||Ax + b||2 <= c.T x
                        Fx = g

    缩写：
        个股i，总数n
        因子j，总数k
        日期t，总数T

    具体实现：
        最小化预期风险：（已实现）
            qp:
                P = (XF[t-1]X[t-1] + sig[t-1] )
        最大化绝对收益：（已实现）
            lp:
                P = -R'                                         以预期收益率  为优化目标
                R'= X[t-1]F[t-1] + spec[t-1].mean()             以预测的因子收益、特定收益  为优化目标
                R'= R' - f(w[t-1])                              考虑调仓
        最小化跟踪误差：（已实现）
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
        禁止做空，没有资金:（已实现）
            s.t.
                A = [1],        b = [1]                         权重和为0
                G = [1],        h = [0]                         禁止卖空
        允许做空：
            s.t.
                G = None,       h = None        允许
        风格/行业中性:（已实现）
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


三、计算模块：

    数据层
        from b_model.barra_multiple_factor.barra_factor import *
    优化器输入层
        from b_model.port_opt.port_opt import Neutral_Port_Opt
    优化器层
        from b_model.port_opt.port_opt import
            min_var_neutral_portfolio
            max_rtn_neutral_portfolio
            min_tracking_error_neutral_portfolio

    数据层：（barra_factor模块的函数、枚举类）
        从数据库读取expo、fac_cov等的逻辑。
        提供 按指数选股/自选股/选择因子类型 多种分支
        多维数据采用双重索引，以方便日后拓展回测模块。
    优化器输入层：（Neutral_Port_Opt 优化器类）
        调用数据层函数，获取优化器类的数据成员
        对齐、去重，去除双重索引
    优化器层：（port_opt 模块的三个方法）
        主要数学逻辑。
        需要确保矩阵参数行列对齐。
        * 形参已经不是凸优化的标准形式。

    * 粒度说明：
        优化器层的参数不是凸优化的标准形式。
        （如二次优化：P,q,G,h,A,b)
        之所以没有将这一块直接整合到优化器输入层，
        是为了分开数据获取部分和矩阵运算部分。


四、改进方向

    丰富数据层函数（get_barra_expo, get_selected_stock_rtn)
    的条件，提供多条件下的SQL语句，以解决只能选取风格因子、
    必须选取基准的问题；
    开发回测模块。只需在优化器输入层加入循环语句即可（循环内，
    用双重索引的level=1 index 取出不含双重索引的DataFrame）。
    主要瓶颈是get_barra_expo，返回的expo矩阵很大，非常耗时。

"""

import warnings
import math
from datetime import timedelta
import numpy as np
import pandas as pd
from pandas import Series
import cvxopt
from cvxopt import solvers
from b_model.barra_multiple_factor.barra_factor import *


class Neutral_Port_Opt(object):
    """
    实现中性组合优化器类，选取一种优化类型，适当传参，即可给出权重序列。
    """

    def __init__(self, t_date: str, benchmark: Index = Index.i000300,
                 factorgroup: FactorGroup = FactorGroup.Indices, max_weight: float = 0.03):
        """
        :param t_date:
        :param benchmark:
        :param factorgroup:
        :param max_weight:
        :return: 优化器类对象，可调用三种优化方法。
        """
        # 必选数据成员：展望基准日，优化组合
        self.t_date = t_date
        self.bench = benchmark
        self.factorgroup = factorgroup
        self.max_weight = max_weight

        # 备用数据成员
        self.neu_fac = []
        self.__non_bench_stock = []
        self.selected_stock_exp_rtn = pd.DataFrame()
        return

    def set_neutral_fac(self, *facs: Factor) -> list:
        """
        调用此方法，以显式地设定需要中性的因子。

        :param facs:
        :return:
        """
        for fac in facs:
            self.neu_fac.append(fac)
        return

    @log
    def set_stock_exp_rtn(self, exp_rtn: DataFrame):
        """
        调用此方法，以显式地提供自选股，以及设定自选股期望收益（可填None）。

        :param exp_rtn: columns = ["ASSETID", "exp_rtn"]，其中期望收益为《年化》。
        :return: DataFrame。
        """
        self.selected_stock_exp_rtn = exp_rtn
        self.bench_list = get_bench_weight(self.t_date, self.t_date, self.bench
            ).columns.tolist()
        for stock in self.selected_stock_exp_rtn.loc[:, "ASSETID"]:
            if stock not in self.bench_list:
                self.__non_bench_stock.append(stock)
        return

    def __bench_wgt_extension(self):
        self.bench_wgt = get_bench_weight(self.t_date, self.t_date, self.bench)
        for stock in self.__non_bench_stock:
            self.bench_wgt[stock] = 0
        return

    """
    下列三个方法，各自调用了优化器。

    惯例：
        取数函数，第一行是函数参数，第二行是去除双重索引、对齐、排序。

        一律在取数的同时，去掉双重索引中的日期
        （单索引的日期无所谓）以防在调用索引对齐矩阵时出错。
        Neutral_Port_Opt类的数据成员，都不含双重索引。
        优化器的输出也不含双重索引。

        一律对齐矩阵、向量。
        虽然np.dot(df1, df2)中，会自动对齐df的行列，
        但若有丢失index/col的处理（如 np.diag(series), df.values），就失去自动对齐功能。
        故统一手动对齐。

        一律用expo的index/col对齐。（去掉双重索引）
        优化器输出（权重）的索引，与expo相同。

    """

    @log
    def get_min_var_port(self, neu_type:Neutral_Type = Neutral_Type.Zero) -> Series:
        """
        求最小方差组合。
        初稿：陈开琛 2018-04-23
        """
        self.expo = get_barra_expo(beg_date=self.t_date, end_date=self.t_date, bench_mark=self.bench, factorgroup=self.factorgroup, stock_list=self.__non_bench_stock
                                   ).loc[self.t_date]
        self.fac_cov = get_barra_cov(self.t_date, self.t_date, self.factorgroup
                                     ).loc[self.t_date].loc[self.expo.columns.tolist(), self.expo.columns.tolist()]
        self.spec_risk = get_barra_spec_risk(self.t_date, self.t_date, self.__non_bench_stock, self.bench
                                             ).loc[:, self.expo.index.tolist()]
        self.__bench_wgt_extension()
        self.bench_wgt = self.bench_wgt.loc[self.t_date, self.expo.index.tolist()]
        # self.bench_wgt 至此成为 Series

        if neu_type == "Zero":
            self.wgt = min_var_neutral_portfolio(cov_mat=self.fac_cov, expo=self.expo, spec_risk=self.spec_risk, neutral_list=self.neu_fac, max_weight=self.max_weight)
        elif neu_type == "Benchmark":
            self.wgt = min_var_neutral_portfolio(cov_mat=self.fac_cov, expo=self.expo, spec_risk=self.spec_risk, neutral_list=self.neu_fac, bench_wgt=self.bench_wgt, max_weight=self.max_weight)
        self.opt_type = "min_var"
        self.__port_statistics()
        return self.wgt

    def get_max_rtn_port(self, exp_month: int, neu_type:Neutral_Type = Neutral_Type.Zero) -> Series:
        """
        求最大收益组合。
        初稿：陈开琛 2018-04-24
        """
        self.expo = get_barra_expo(self.t_date, self.t_date, self.bench, self.factorgroup, self.__non_bench_stock
                                   ).loc[self.t_date]
        self.fac_cov = get_barra_cov(self.t_date, self.t_date, self.factorgroup
                                     ).loc[self.t_date].loc[self.expo.columns.tolist(), self.expo.columns.tolist()]
        self.spec_risk = get_barra_spec_risk(self.t_date, self.t_date, self.__non_bench_stock, self.bench
                                             ).loc[:, self.expo.index.tolist()]
        self.t_date_end = (pd.to_datetime(self.t_date) + timedelta(days = exp_month*30)).strftime("%Y-%m-%d")
        print("self.__non_bench_stock", self.__non_bench_stock)
        self.exp_rtn = get_selected_stock_rtn(self.t_date, self.t_date_end, Period.DAY, self.__non_bench_stock, self.bench
                                              ).pivot_table(values='RTN', index='T_DATE', columns="ASSETID"
            ).fillna(0).mean().loc[self.expo.index.tolist()]
        if self.selected_stock_exp_rtn.empty != True:
            for i, row in self.selected_stock_exp_rtn.iterrows():
                self.exp_rtn.loc[row["ASSETID"]] = row["exp_rtn"] * exp_month / 12
        self.__bench_wgt_extension()
        self.bench_wgt = self.bench_wgt.loc[self.t_date, self.expo.index.tolist()]
        # self.bench_wgt 至此成为 Series

        if neu_type == "Zero":
            self.wgt = max_rtn_neutral_portfolio(exp_rtn=self.exp_rtn, expo=self.expo, neutral_list=self.neu_fac, max_weight=self.max_weight)
        elif neu_type == "Benchmark":
            self.wgt = max_rtn_neutral_portfolio(exp_rtn=self.exp_rtn, expo=self.expo, neutral_list=self.neu_fac, bench_wgt=self.bench_wgt, max_weight=self.max_weight)
        self.opt_type = "max_rtn"
        self.__port_statistics()
        return self.wgt

    def get_min_te_port(self, neu_type:Neutral_Type = Neutral_Type.Zero) -> Series:
        """
        求最小跟踪误差组合。
        初稿：陈开琛 2018-04-23
        """
        self.expo = get_barra_expo(self.t_date, self.t_date, self.bench, self.factorgroup, self.__non_bench_stock
                                   ).loc[self.t_date]
        self.fac_cov = get_barra_cov(self.t_date, self.t_date, self.factorgroup
                                     ).loc[self.t_date].loc[self.expo.columns.tolist(), self.expo.columns.tolist()]
        self.spec_risk = get_barra_spec_risk(self.t_date, self.t_date, self.__non_bench_stock, self.bench
                                             ).loc[:, self.expo.index.tolist()]
        self.__bench_wgt_extension()
        self.bench_wgt = self.bench_wgt.loc[self.t_date, self.expo.index.tolist()]
        # self.bench_wgt 至此成为 Series

        if neu_type == "Zero":
            self.wgt = min_tracking_error_neutral_portfolio(cov_mat=self.fac_cov, expo=self.expo, spec_risk=self.spec_risk, neutral_list=self.neu_fac, bench_wgt=self.bench_wgt, max_weight=self.max_weight)
        elif neu_type == "Benchmark":
            self.wgt = min_tracking_error_neutral_portfolio(cov_mat=self.fac_cov, expo=self.expo, spec_risk=self.spec_risk, neutral_list=self.neu_fac, bench_wgt=self.bench_wgt, neu_type=neu_type, max_weight=self.max_weight)
        self.opt_type = "min_te"
        self.__port_statistics()
        return self.wgt

    def __port_statistics(self):
        """
        计算指标，以评估优化效果。指标包括：
            指定的因子暴露,与bench比较  port_bench_expo
            前10大重仓股                top_holding
            预期风险                    exp_risk
            预期收益                    exp_rtn
            预期跟踪误差                port_te
        """

        self.P = np.dot(self.expo, np.dot(self.fac_cov, self.expo.T)) + np.diag(self.spec_risk.values.flat)

        if self.opt_type == "min_te":
            # 预期跟踪误差（年化标准差）
            self.port_te = math.sqrt(np.dot(self.wgt.T - self.bench_wgt.T, np.dot( self.P, self.wgt - self.bench_wgt)))

        else:
            if self.opt_type == "max_rtn":
                # 组合预期收益
                self.exp_rtn = pd.DataFrame({"port": np.dot(self.wgt, self.exp_rtn),
                                             "bench": np.dot(self.bench_wgt, self.exp_rtn)},
                                            index = ["exp_rtn"])

        # 指定因子暴露,与bench比较
        self.port_bench_expo = pd.DataFrame({
            "port":     np.dot( self.expo.T, self.wgt),
            "bench":    np.dot( self.expo.T, self.bench_wgt)
        }, index = self.expo.columns
        ).sort_values("port", ascending=True)

        # 组合预期风险（年化标准差）
        self.exp_risk = pd.DataFrame({"port": math.sqrt(np.dot(self.wgt.T, np.dot( self.P, self.wgt))),
                                      "bench": math.sqrt(np.dot(self.bench_wgt.T, np.dot( self.P, self.bench_wgt)))},
                                     index = ["exp_risk"])

        # 前10大重仓股
        self.top_holding = self.wgt.sort_values(ascending=False).head(10)

        return


@log
def min_var_neutral_portfolio(cov_mat: DataFrame, expo: DataFrame, spec_risk: DataFrame, neutral_list: list, bench_wgt = 0,
                              max_weight: float = 0.03) -> Series:
    """

    :param bench_wgt:
    :param cov_mat:
    :param expo:
    :param neutral_list:
    :param max_weight:
    :return: Series。索引与expo的相同。
    """
    n, k = expo.shape

    spec_risk = np.diag(
        spec_risk.loc[:, expo.index.tolist()].values.flat
    )

    P = cvxopt.matrix( (np.dot(expo,
                              np.dot(cov_mat, expo.T) )
                       + spec_risk).astype(np.float64))    # n * n
    q = cvxopt.matrix(0.0, (n, 1))
    print("Neutral include:", neutral_list, "P size:", P.size)

    G = -np.identity(n)
    h = np.zeros((n, 1))

    G = np.vstack((G,
                   np.identity(n)))
    h = np.vstack((h,
                   np.full((n, 1), max_weight)))

    for neu_fac in neutral_list:
        if type(bench_wgt) != int:
            neu_fac_expo = np.dot(expo[neu_fac], bench_wgt)
        else:
            neu_fac_expo = 0

        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            neu_fac_expo+0.0005,
            -(neu_fac_expo)+0.0005  ))
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
def max_rtn_neutral_portfolio(exp_rtn: Series, expo: DataFrame, neutral_list: list, bench_wgt = 0, max_weight: float = 0.03) -> Series:

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
        if isinstance(bench_wgt, pd.Series):
            neu_fac_expo = np.dot(expo[neu_fac], bench_wgt)
        else:
            neu_fac_expo = 0

        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            neu_fac_expo+0.0005,
            -(neu_fac_expo)+0.0005  ))

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
def min_tracking_error_neutral_portfolio(cov_mat: DataFrame, expo: DataFrame, spec_risk: DataFrame,
        neutral_list: list, bench_wgt: Series, neu_type: Neutral_Type = Neutral_Type.Zero, max_weight: float = 0.03) -> Series:

    n, k = expo.shape

    spec_risk = np.diag(
        spec_risk.loc[:, expo.index.tolist()].values.flat
    )

    P = cvxopt.matrix( np.dot(expo, np.dot(cov_mat, expo.T) ) + spec_risk.astype(np.float64))    # n * n
    # P = cvxopt.matrix(np.dot(expo, np.dot(cov_mat, expo.T)).astype(np.float64))    # n * n
    q = cvxopt.matrix(-np.dot(P, bench_wgt.T))
    print("Neutral include:", neutral_list, "P size:", P.size)

    G = -np.identity(n)
    h = np.zeros((n, 1))

    G = np.vstack((G,
                   np.identity(n)))
    h = np.vstack((h,
                   np.full((n, 1), max_weight)))

    for neu_fac in neutral_list:
        if neu_type == Neutral_Type.Zero:
            neu_fac_expo = 0
        elif neu_type == Neutral_Type.Benchmark:
            neu_fac_expo = np.dot(expo[neu_fac], bench_wgt)
        G = np.vstack((     G,
            expo[neu_fac].values,
            -expo[neu_fac].values ))
        h = np.vstack((     h,
            neu_fac_expo+0.0005,
            -(neu_fac_expo)+0.0005  ))

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


def to_do():
    """"""
    """
    日志
    0425
        加入特异风险
        对齐
        处理非权重股的权重
        出个评价模块
        统一双重索引
        允许对基准风格中性

    0426
        产品运行
            max_rtn：能否不设exp_rtn？exp_rtn统一用年化？
        注释
        测试剩余两个
        排版
    """
    return
