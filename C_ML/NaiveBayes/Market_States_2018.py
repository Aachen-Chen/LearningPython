

import time
import datetime
import copy
from math import sqrt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from A_Infrastructure.common import log


@log
def mock():
    # hl, window, t1, t2, te = 10, 25, 30, 10, 10   # half life, ewma window, ex post, ex ante
    # hl, window, t1, t2, te = 10, 25, 300, 100, 100   # half life, ewma window, ex post, ex ante
    # hl, window, t1, t2, te = 100, 250, 3000, 1000, 1000   # half life, ewma window, ex post, ex ante
    hl, window, t1, t2, te = 100, 250, 6000, 1500, 1500  # half life, ewma window, ex post, ex ante
    # hl, window, t1, t2, te = 100, 250, 10000, 10000, 1000   # half life, ewma window, ex post, ex ante
    Total, T = sum([window - 1 + 5, t1, t2, te]), sum([t1, t2, te])

    seed = 32
    np.random.seed(seed)
    d = pd.DataFrame(np.round(np.random.rand(T+5)*5-2.5), columns=['t'])
    print(d['t'].value_counts(sort=False, dropna=False).sort_index(ascending=False))
    return d

@log
def mock2(numbers, rand):
    df = pd.DataFrame(columns=['t'])
    for i in range(numbers):
        if i%5 == 0:
            df.loc[i] = np.round(np.random.rand()*2-1.0) if rand else 0.
        elif i%5 == 1:
            df.loc[i] = np.round(np.random.rand()*2) if rand else 1.
        elif i % 5 == 2:
            df.loc[i] = 2.
        elif i % 5 == 3:
            df.loc[i] = -2.
        else:
            df.loc[i] = np.round(np.random.rand()* -2) if rand else -1.
    print(df['t'].value_counts(sort=False, dropna=False).sort_index(ascending=False))
    return df


@log
def read(location) -> DataFrame:
    """"""
    """    ["t_date", "rtn"]    """
    d = pd.read_excel(location, header=None, sheet_name=0)
    # print(d.head(2))
    d = pd.DataFrame({
        "t_date": pd.to_datetime(d[0]),
        "rtn": d[1] / 100
    })
    # print("Total length:", len(d))
    # d = d.loc[(d['rtn']!=0.0),:]
    # print("Trade day:", len(d))
    d.set_index("t_date", inplace=True)
    """    ["t_date"], ["rtn"]    """
    return d


@log
def select_hour(d: pd.DataFrame) -> DataFrame:
    """"""
    """    ["t_date_hour"], ["rtn"]    """
    print(d.head(2))
    d = d.between_time(datetime.time(15, 00, 00), datetime.time(15, 00, 00),
                       include_start=True, include_end=True)
    print(d.head())
    # d.reindex(d.index.date.tolist(), inplace=True)
    d.index = d.index.date
    print(d.head())
    """    ["t_date"], ["rtn"]    """
    return d

@log
def ewma(d: DataFrame, halflife: int, span: int, col:str = "rtn", nothing=True) -> DataFrame:
    """"""
    """    ["t_date"], ["rtn"]    """
    # print(d.head(2))

    if nothing:
        d = pd.DataFrame({
            col: d[col], "ewma": np.zeros(len(d)), "t": d[col]
        }, index=d.index)
    else:
        lam = pow(0.5, 1 / halflife)
        dd = pd.DataFrame({"ewma": np.zeros(len(d))}, index=d.index)
        # d.loc[:, 'ewma'] = 0
        for i, t in enumerate(d.index.tolist()[span: len(d)]):
            sigma = 0
            for j, th in enumerate(d.index.tolist()[i: i + span]):
                if j == 0:
                    sigma = pow(d.loc[th, col], 2)
                else:
                    sigma = lam * sigma
                    sigma += (1-lam) * pow(d.loc[th, col], 2)
            dd.loc[t, 'ewma'] = sqrt(sigma)
        # d.loc[:, 't'] = d.loc[:, col] / d.loc[:, "ewma"]
        d = pd.DataFrame({
            col: d[col], "ewma": dd['ewma'], "t": d[col] / dd['ewma']
        }, index=d.index)
    """   [col, "ewma", "t"]  """
    return d.loc[:, [col, "ewma", "t"]].iloc[span: , :]

@log
def categorize(d: DataFrame, num_states: int = 5,
               col: str = "t", types:str="normal") -> DataFrame:
    """"""
    """    ["t_date"], ["rtn", "ewma", 't']    """
    # print(d.head(2))
    # d.loc[:, col].dropna(inplace=True)
    if num_states == 5:
        states = [-2, -1, 0, 1, 2]
    elif num_states == 2:
        states = [0, 1]
    else:
        raise ValueError("Only 2 or 5 states supported.")

    kth = np.linspace(0.0, 1.0, num_states + 1)     # 0.2, 0.4, 0.6, 0.8
    if types == "normal":
        quantiles = stats.norm.ppf(kth[1:-1], d[col].mean(), d[col].std())
    elif types == "even":
        quantiles= []
        for i, k in enumerate(kth[1:-1]):
            # print(int(k*len(d)))
            quantiles.append(d[col].sort_values().iloc[int(k*len(d))])
    print(d[col].mean(), d[col].std())

    for i in range(num_states - 1):
        if i == 0:
            d['state'] = np.where(d[col] >= quantiles[i], states[i + 1], states[i])
            # print(d)
        elif i > 0:
            d['state'] = np.where(d[col] >= quantiles[i], states[i + 1], d['state'])

    d = pd.DataFrame({
        "rtn" : d['rtn'], "t" : d['state'], "ewma": d['ewma']
    }, index= d.index)
    print(d['t'].value_counts())
    """ ["t_date"], ["rtn", "t", 'ewma'] """
    return d

@log
def d_making(location: str = r"D:\data\002068.xlsx", num_states:int=5,
             type:str="small", category:str='even', without_ewma=True):
    print(location)
    d = read(location)
    # h = read(location)

    if type =='small':
        show = 250
        n, hl = 20, 4
    else:
        show = len(d)
        n, hl = 252, 42

    d = ewma(d.iloc[:show,:], hl, n, 'rtn', nothing=without_ewma).iloc[n:, :]
    # h = ewma(h.iloc[:show,:], hl, n, 'rtn').iloc[n:, :]

    d = categorize(d, num_states, 't', category)
    # h = categorize(h, 5, 't')

    # d = pd.DataFrame({ "rtn": d["rtn"], "t": d["t"], "t-h": h["t"] }, index=d.index)
    # """  ["rtn", "t", "t-h"]    """
    d = pd.DataFrame({
        "rtn": d["rtn"], "t": d["t"], "ewma": d['ewma']
    }, index=d.index)
    """  ["rtn", "t", 'ewma']    """
    return d

@log
def d_making2(d:DataFrame, num_states:int=5,
             type:str="small", category:str='even', without_ewma=True):
    if type =='small':
        show, n, hl = 250, 20, 4
    else:
        show, n, hl = len(d), 252, 42

    d = ewma(d.iloc[:show,:], hl, n, 'rtn', nothing=without_ewma).iloc[n:, :]
    d = categorize(d, num_states, 't', category)

    d = pd.DataFrame({"rtn": d["rtn"], "t": d["t"], "ewma": d['ewma']
        }, index=d.index)
    """  ["rtn", "t", 'ewma']    """
    return d


@log
def training_set(d: DataFrame):
    """"""
    """  ["rtn", "t", "t-h"]    """
    # print(d.head(2))
    for i in range(1, 6):
        d.loc[:, 't-' + str(i)] = d['t'].shift(i)
    # print("d_states.head(10), before na\n", d.head(1))
    d.dropna(inplace=True)
    # print("d_states.head(10)\n", d.head(1))
    # d = d.iloc[:, ::-1]
    # print("d_states.head(10)\n", d.head(3))
    """  ["rtn", "t", "t-h", 't-5', 't-4', 't-3', 't-2', 't-1', 'ewma']    """
    return d

@log
def predicts(d: DataFrame, t12:tuple=(0.4,0.4)):
    # print(d.head(1))
    t1, t2, te = round(len(d) * t12[0]), round(len(d) * t12[1]), round(len(d) * (1-sum(t12)))
    T = sum([t1, t2, te])
    tag_5 = ['t-5', 't-4', 't-3', 't-2', 't-1']
    tag_5r = ['t-5', 't-4', 't-3', 't-2', 't-1', 't_r']
    # tag_5 = ['t-5', 't-4', 't-3', 't-2', 't-1', 't-h']
    # tag_5r = ['t-5', 't-4', 't-3', 't-2', 't-1', 't-h', 't_r']

    d_l = d.index.tolist()
    t1_l = d_l[:t1]
    t2_l = d_l[t1: t1 + t2]
    te_l = d_l[t1 + t2: t1 + t2 + te]

    d['pred'] = 3
    # print(d.head(1))
    """ 't-5', 't-4', 't-3', 't-2', 't-1', 't', 'pred=3' """
    print("----------------train 1--------------")

    for i, t in enumerate(t2_l):
        d_ta = d.loc[d_l[i:i + t1], :]
        """ 停牌期间也训练 """
        x_ta = d_ta.loc[:, tag_5]
        y_ta = d_ta.loc[:, 't']
        """ 改进：停牌期间不作训练 """
        # x_ta = d_ta.loc[(d_ta['rtn']!=0), tag_5]
        # y_ta = d_ta.loc[(d_ta['rtn']!=0), 't']

        # print(len(d_ta), len(x_ta))
        model = GaussianNB().fit(
            x_ta.values,
            y_ta.values.ravel()
        )
        d.loc[t, "pred"] = model.predict(d.loc[t, tag_5].values.reshape(1, -1))
    print("End of train_1")

    """ pred：   t2的都为1/0，其他区域还是3 """
    """ t_r：    t2的都为1/0，其他区域还是0 """
    """ t2_r：   都是3 """
    d['t_r'] = 0.
    d.loc[(d["pred"] == d["t"]), "t_r"] = 1.
    d["t2_r"] = 3.

    d2 = d.loc[t2_l, :]
    # print("t1_r\n", d2.loc[(d2['rtn']!=0), 't_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False))
    # print(d.head(1))
    # print(d.tail(1))
    """ 't-5', 't-4', 't-3', 't-2', 't-1', 't_r', 't', 'pred', 't2_r' """

    print("----------------train 2--------------")
    for i, t in enumerate(te_l):
        d_ta = d.loc[d_l[i + t1 : i + t1 + t2], :]
        """ 停牌期间也训练 """
        # x_ta = d_ta.loc[:, tag_5r]
        # y_ta = d_ta.loc[:, ['t']]
        """ 改进：停牌期间不作训练 """
        x_ta = d_ta.loc[(d_ta['rtn']!=0), tag_5r]
        y_ta = d_ta.loc[(d_ta['rtn']!=0), ['t']]
        # print(len(d_ta), len(x_ta))

        model_origin = GaussianNB().fit(
            x_ta.loc[:, tag_5].values,
            y_ta.values.ravel() )
        x_test = copy.deepcopy(d.loc[t, tag_5r])
        d.loc[t, 'pred'] = model_origin.predict( x_test.loc[tag_5].values.reshape(1, -1))
        d.loc[t, "t_r"] = 1 if d.loc[t, "pred"] == d.loc[t, 't'] else 0

        """ train with assumed result """
        model_with_result = GaussianNB().fit(
            x_ta.values,
            y_ta.values.ravel())
        x_test.loc['t_r'] = 1.
        # print("--1--\n", x_test.loc['t_r'])
        pred_1 = model_with_result.predict(x_test.values.reshape(1, -1))
        # print(d.loc[t, "pred"], pred_1)
        if d.loc[t, "pred"] != pred_1:
            # x_test.loc['t_r'] = 0.
            # print("--0--\n", x_test.loc['t_r'])
            # pred_2 = model_with_result.predict(x_test.values.reshape(1, -1))
            pred_2 = 0 if pred_1 == 1 else 1
            d.loc[t, "pred"] = pred_2
            # print(d.loc[t, "pred"], pred_2)
        d.loc[t, "t2_r"] = 1 if d.loc[t, "pred"] == d.loc[t, 't'] else 0

        # """ train with only correct prediction """
        # model_correct = GaussianNB().fit(
        #     x_train.loc[x_train['t_r']==1, tag_5].values,
        #     y_train.loc[x_train['t_r']==1, :].values.ravel() )
        # d.loc[t, "pred"] = model_correct.predict(x_test.loc[tag_5].values.reshape(1, -1))
        # d.loc[t, "t2_r"] = 1 if d.loc[t, "pred"] == d.loc[t, 't'] else 0

    print("End of train_2")

    # print(d.head(1), '\n')
    # print(d.tail(1))
    print(pd.DataFrame({
        "d['t']":   d['t'].value_counts(normalize=True),
        "d['pred']":d.loc[d_l[t1:t1+t2+te],'pred'].value_counts(normalize=True)
    }))
    print("---------------- Precision --------------")
    d2 = d.loc[t2_l, :]
    de = d.loc[te_l, :]
    precision = pd.DataFrame({
        "t1_r":d2.loc[(d2['rtn']!=0), 't_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False),
        "t2_r":de.loc[(de['rtn']!=0), 't2_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False),
        "t2_r_origin":de.loc[(de['rtn']!=0), 't_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False)
    })
    # print("t1_r\n", d2.loc[(d2['rtn']!=0), 't_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False))
    # print("t2_r\n", de.loc[(de['rtn']!=0), 't2_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False))
    # print("t2_r_origin\n", de.loc[(de['rtn']!=0), 't_r'].value_counts(normalize=True, sort=False).sort_index(ascending=False))
    print(precision)
    # print("total\n",d['t_r'].value_counts(sort=False, dropna=False).sort_index(ascending=False))

    """
    """
    print(t1, t2, te, len(d))
    print(t1_l[-1], t2_l[-1], te_l[-1])
    # import cvxopt.solvers as optsolvers
    # optsolvers.options['show_progress'] = False
    # sol = optsolvers.qp(t1, t2, te, d_l, d2, de)

    # from pandas import ExcelWriter
    # writer = ExcelWriter('D:\data\cache.xlsx')
    # d.to_excel(writer, 'Sheet1')
    """  ["rtn", "t", "t-h", 't-5', 't-4', 't-3', 't-2', 't-1', 'ewma']    """
    """ 新增：["pred", "t_r", "t2_r"] """
    return d, precision

@log
def trade(d: DataFrame,
          ret:str='rtn', sig:str='pred',
          buy:tuple=(2,), hold:tuple=(1,),
          stop_loss:float=0.05, s:float=0.001)->DataFrame:
          # )->tuple:
    """"""
    # print(d.head(2))

    """ trading """
    dd = pd.DataFrame({
        "posi":     np.zeros(len(d.loc[d['pred']!=3,:])),
        # "posi_chg": np.zeros(len(d.loc[d['pred']!=3,:])),
        "p_rtn":    np.zeros(len(d.loc[d['pred']!=3,:])),
        "p_cum":    np.zeros(len(d.loc[d['pred']!=3,:])),
        "dd":       np.zeros(len(d.loc[d['pred']!=3,:])) }, index=d.loc[d['pred']!=3,:].index)
    dd['rtn'] = d['rtn']
    """ dd.columns.tolist() = ["posi", "posi_chg", "p_rtn", "p_cum", "dd", "rtn"] """
    d_l = dd.index.tolist()

    for i, td in enumerate(d_l):
        """ pre_trade data """
        #     t-1 cum
        #     t-1 dd
        """ pred -> buy/sell -> position """
        if d.loc[td, 'pred'] in buy:
            dd.loc[td, 'posi'] = 1
        elif d.loc[td, 'pred'] in hold: # 若没有hold状态，hold=[]即可。
            dd.loc[td, 'posi'] = dd.loc[d_l[i - 1], 'posi']

        if i > 0:
            if dd.loc[d_l[i-1], 'dd'] < -stop_loss:     # 这一条决定了不能直接列运算，必须每日迭代。
                dd.loc[td, 'posi'] = 0
        #     else:
        #         dd.loc[td, 'posi'] = dd.loc[td, 'posi'] \
        #             * d.loc[td, 'ewma'] / d.loc[d_l[i-1], 'ewma']
        #     dd[td, 'posi_chg'] = dd.loc[td, 'posi'] - dd.loc[d_l[i-1], 'posi']
        # else:
        #     dd[td, 'posi_chg'] = dd.loc[td, 'posi']
        #
        # """ cum_rtn (0 base) """
        # if i > 0:
        #     if dd[td, 'posi_chg'] < 0:
        #         dd.loc[td, 'p_cum'] = \
        #             (dd.loc[d_l[i - 1], 'p_cum'] + 1) * \
        #             abs(dd[td, 'posi_chg']) * \
        #             (1-s) - 1
        #     # 先减了cum_rtn，会导致后面的stock总额也减少。不过减少不多，而且在1-0仓位状态下，没有影响。
        #         dd.loc[td, 'p_cum'] = \
        #             (d.loc[td, 'rtn'] * dd.loc[td, 'posi']+1) * \
        #             (dd.loc[d_l[i - 1], 'p_cum'] + 1) - 1
        #   2018-05-02 暂缓开发
        dd.loc[td, 'p_rtn'] = d.loc[td, 'rtn'] * dd.loc[td, 'posi']

        """ drawdown """
        if dd.loc[td, 'p_rtn'] >= 0:
            dd.loc[td, 'dd'] = 0
        else:
            if i > 0:
                dd.loc[td, 'dd'] = (dd.loc[td, 'p_rtn']+1) * (dd.loc[d_l[i-1], 'dd']+1) -1
            else:
                dd.loc[td, 'dd'] = dd.loc[td, 'p_rtn']

        """ cum_rtn"""
        """ ！！以0为基准，以兼容回撤计算！！ """
        if i > 0:
            posi_chg = dd.loc[td, 'posi'] - dd.loc[d_l[i - 1], 'posi']
            if posi_chg < 0:
                """ 印花税 """
                dd.loc[td, 'p_cum'] = (1-s)*(1 + dd.loc[d_l[i - 1], 'p_cum']) * (1 + dd.loc[td, 'p_rtn']) -1
            else:
                dd.loc[td, 'p_cum'] = (1 + dd.loc[d_l[i - 1], 'p_cum']) * (1 + dd.loc[td, 'p_rtn'])-1
        else:
            dd.loc[td, 'p_cum'] = dd.loc[td, 'p_rtn']

    d['p_rtn'] = dd['p_rtn']
    # dd = dd.apply(lambda x: np.cumprod(x+1)-1)
    d['b_cum'] = np.cumprod(dd['rtn']+1)-1
    # d['b_cum'] = dd['rtn']
    d['p_cum'] = dd['p_cum']
    # d['p_cum'] = dd['p_rtn']

    """ 测评 """
    # dd = pd.DataFrame({
    #     # "posi" : dd['posi'] / 10,
    #     "stand": d['rtn']/d['ewma'],
    #     "state": d['t']/10+0.2,
    #     "pred" : d['pred']/10+0.2,
    #     "b_cum": d['b_cum'],
    #     "p_cum": d['p_cum']
    # }, index=d.index)
    # dd = dd.loc[d['pred']!=3, : ]
    # dd.loc[:,'stand'] = dd['stand']/max((d['rtn']/d['ewma']).tolist())

    # print(d.head(1))
    # print(d.tail(1))
    return d    # , dd

@log
def raroc(s:Series)->Series:
    print(" ")
    return s

@log
def statistic(s:Series, p, freq:str='D', r_type:str='rtn'):
    print(s.name, s.iloc[1])
    if isinstance(p, tuple):
        t2 = p[1]/(1-p[0])
        kth = [0.0, t2/2, t2, t2+0.5*(1-t2), 1.0]
        p=len(p)*2
    elif isinstance(p, int):
        kth = np.linspace(0.0, 1.0, p + 1)           # p+1个点
    kth = [round((len(s)-1)*x) for x in kth]    # iloc
    lth = s.iloc[kth].index.tolist()            # loc
    sta=pd.DataFrame(np.zeros((p, 4)),
        columns= ["rtn", "std", "sharpe", "max_dd"],
        index = lth[1:])
    print(lth[0])
    # print("--1----", s.name, s.iloc[27])

    if r_type == 'rtn':
        # s.plot()
        for i in range(p):
            max_dd = cur_drawdown(pd.DataFrame({"rtn": s.loc[lth[i]:lth[i + 1]]}),
                "rtn", r_type=r_type)['dd'].min()
            rtn, std = s.loc[lth[i]:lth[i + 1]].mean(), s.loc[lth[i]:lth[i + 1]].std()
            if freq=='D':
                rtn, std = (rtn+1) ** 252. -1 , std * sqrt(252)
            elif freq=='W':
                rtn, std = (rtn+1) ** 52. -1 , std * sqrt(52)
            elif freq=='M':
                rtn, std = (rtn+1) ** 12. -1 , std * sqrt(12)
            sta.loc[lth[i + 1], :] = [rtn, std, rtn / std, max_dd]
    elif r_type == 'cum':
        """万恶之源pd.Series(s.values)，没用的！还是原来的对象！"""
        ss= copy.deepcopy(s)
        s_l= s.index.tolist()
        for i, t in enumerate(s_l):
            # print("i", i)
            if i > 0:
                # print("cal:", (s.loc[t] + 1), (s.loc[s_l[i - 1]] + 1), (s.loc[t] + 1) / (s.loc[s_l[i - 1]] + 1) - 1)
                ss.loc[t] = (s.loc[t] + 1) / (s.loc[s_l[i - 1]] + 1) - 1
                # print("ss", ss.loc[t])
            elif i == 0:
                ss.loc[t] = s.loc[t]
        s = pd.Series(ss.values, index=s.index, name=s.name)
        for i in range(p):
            max_dd = cur_drawdown(pd.DataFrame({"rtn": s.loc[lth[i]:lth[i + 1]]}),
                                  "rtn", r_type='rtn')['dd'].min()
            rtn, std = s.loc[lth[i]:lth[i + 1]].mean(), s.loc[lth[i]:lth[i + 1]].std()
            if freq == 'D':
                rtn, std = (rtn + 1) ** 252. - 1, std * sqrt(252)
            elif freq == 'W':
                rtn, std = (rtn + 1) ** 52. - 1, std * sqrt(52)
            elif freq == 'M':
                rtn, std = (rtn + 1) ** 12. - 1, std * sqrt(12)
            sta.loc[lth[i + 1], :] = [rtn, std, rtn / std, max_dd]
    print(sta)
    return


@log
def drawing(dd:DataFrame, pt:float=0.6):
    print(dd.head(2))
    d_l = dd.index.tolist()
    # b1 = d_l.index('2016-09-01')
    b1 = round(pt*len(d_l))
    b2 = round((pt+0.1) * len(d_l))
    b3 = round((pt+0.2) * len(d_l))
    b4 = round((pt + 0.3) * len(d_l))
    dd.loc[(dd['pred']!=3), ['b_cum', 'p_cum']].plot()
    dd['xxx'] = np.cumprod(dd.loc[:,'b_cum']+1)-1
    s = (dd.loc[(dd['pred']!=3), 'pred']/10+0.2)
    s.plot()
    # dd.iloc[b1: b2, :].plot()
    # dd.iloc[b2: b3, :].plot()
    # dd.iloc[b3: b4, :].plot()
    return

def cur_drawdown(d:DataFrame, col:str="rtn", r_type:str='rtn')-> DataFrame:
    d_l = d.index.tolist()
    dd=pd.DataFrame({"dd": np.zeros(len(d_l))}, index=d_l)
    if r_type == 'rtn':
        for i, t in enumerate(d_l):
            if d.loc[t, col] > 0:
                dd.loc[t, "dd"] = 0
            elif d.loc[t, col] < 0:
                if i > 0:
                    dd.loc[t, "dd"] = (d.loc[t, col]+1) * (dd.loc[d_l[i-1], "dd"]+1) -1
                else:
                    dd.loc[t, "dd"] = d.loc[t, col]
    elif r_type =='cum':
        for i, t in enumerate(d_l):
            if d.loc[t, col] >= d.loc[d_l[i-1], col]:
                dd.loc[t, "dd"] = 0
            elif d.loc[t, col] < d.loc[d_l[i-1], col]:
                if i > 0:
                    dd.loc[t, "dd"] = (d.loc[t, col]+1) / (d.loc[d_l[i-1], col]+1)\
                        * (dd.loc[d_l[i-1], "dd"]+1) -1
                else:
                    dd.loc[t, "dd"] = d.loc[t, col]
    d["dd"] = dd["dd"]
    return d

@log
def everything(n:int=1, type:str='small'):
    line = [r"D:\data\000001.xlsx", r"D:\data\002068.xlsx",
            r"D:\data\002070.xlsx", r"D:\data\002109.xlsx",
            r"D:\data\002182.xlsx", r"D:\data\601666.xlsx",
            r"D:\data\601699.xlsx", r"D:\data\000300.xlsx",
            r"D:\data\000007.xlsx", r"D:\data\if00.xlsx",
            r"D:\data\000852.xlsx", r"D:\data\930689.xlsx",
            r"D:\data\930606.xlsx"]
    print(line[n])
    d = d_making(line[n], 2, type, category='even', without_ewma=False)
    d = training_set(d)
    d = predicts(d, (0.4, 0.3))
    # d, dd = trade(d, buy=(2,1), hold=(0,), stop_loss=0.03)
    d = trade(d, buy=(1, ), hold=(1,), stop_loss=0.05, s=0.0)
    drawing(d, 0.2)
    statistic(d.loc[d['pred'] != 3, 'rtn'].dropna(), (0.4, 0.3), 'D', r_type='rtn')    # p可以是4
    statistic(d.loc[d['pred'] != 3, 'p_cum'].dropna(), (0.4, 0.3), 'D', r_type='cum')
    plt.title(line[n])

@log
def anything(n:int=1, type:str='small'):
    from common import read_multi
    time_start = time.time()
    location= r"D:\data\csi500a.xlsx"
    print(location)
    b_rtn02 = read_multi(location).iloc[:, :n]
    c, l=b_rtn02.columns.tolist(), b_rtn02.index.tolist()
    p_rtn02 = pd.DataFrame(np.zeros((len(l), len(c))),
        columns=b_rtn02.columns, index=b_rtn02.index)
    last, time_list = pd.to_datetime('1996-10-26'), []
    precision = pd.DataFrame(np.zeros((len(c), 3)),
        columns=["t1_r","t2_r","t2_r_origin"], index=c)
    """ 这里开始迭代 """
    for i, tag in enumerate(c):
        """    ["t_date"], ["rtn"]    """
        print("*************", i+1, tag, "*************")
        time_list = time_list+[time.time()-time_start]
        print(time_list)
        # print("Total time:", time.time()-time_start, "last round:", )
        d = pd.DataFrame({"rtn": b_rtn02[tag].values}, index=l)
        d = d_making2(d, 2, type, category='even', without_ewma=False)
        d = training_set(d)
        d, precise = predicts(d, (0.4, 0.3))
        d = trade(d, buy=(1, ), hold=(1,), stop_loss=0.05, s=0.0)
        # drawing(d, 0.2)
        p_rtn02.loc[:, tag] = d['p_rtn']
        # 选最后一个上市的日期，开始一起投资
        last = max([last, max(d.loc[(d['pred']==3),:].index.tolist()) ])
        precision.loc[tag,:] = precise.iloc[0, :]
    # output_0502.xlsx
    writer = pd.ExcelWriter(r"D:\data\output_csi500a.xlsx")
    p_rtn02.to_excel(writer, 'Sheet1')
    precision.to_excel(writer, 'Sheet2')
    writer.save()
    b_cum02 = np.cumprod(b_rtn02.loc[l[l.index(last)+1]:, c]+1)-1
    p_cum02 = np.cumprod(p_rtn02.loc[l[l.index(last)+1]:, c]+1)-1
    pb = pd.DataFrame({
        "b_rtn": b_cum02.loc[:, :].sum(axis=1),
        "p_rtn": p_cum02.loc[:, :].sum(axis=1),
    })
    print(pb.index.tolist()[0], pb.index.tolist()[-1])
    pb.plot()

    # drawing(d, 0.2)
    # statistic(d.loc[d['pred'] != 3, 'rtn'].dropna(), (0.4, 0.3), 'D', r_type='rtn')    # p可以是4
    # statistic(d.loc[d['pred'] != 3, 'p_cum'].dropna(), (0.4, 0.3), 'D', r_type='cum')
    # plt.title(line[n])

# for i in range(4):

# everything(1, 'big')
anything(101, 'big')
plt.show()

# d = cur_drawdown(read(line[6]))
# d = pd.DataFrame({'cum': np.cumprod(d['rtn']+1)-1, 'dd': d['dd']}, index=d.index)
# d.iloc[:20,:].plot()


