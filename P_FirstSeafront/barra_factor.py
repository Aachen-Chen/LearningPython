"""
本模块提供Barra相关的函数接口。

修改历史：
    初稿 陈开琛 2018-03-30
    last updated 陈开琛 2018-04-26

符号：
    个股：i，总数n
    因子：j，总数k
    时间：t，总数T

惯例：
    采用Barra_CHN_5S。收益、风险（年化）均为月数据。
    全部返回DataFrame格式；
    DataFrame的列名，均为大写。
    因子名称为列名（如有）。i.e.
        因子暴露矩阵 F([t, n] * k)
        因子回报矩阵 f(t * k)
"""

"""
注意：
    凡是从数据库获取的数据表的表头，都是大写字母。
"""

import time
import pandas as pd
from pandas import DataFrame
from enum import Enum, unique
# from a_infrastructure.dao import execute_sql
# from b_model.common import Index, Period

def log(func):
    def wrapper(*args, **kw):
        print("Call: %s():" % func.__name__)
        return func(*args, **kw)

    return wrapper

@unique
class Neutral_Type(Enum):
    Enum.Zero = "Zero"
    Enum.Benchmark = "Benchmark"

@unique
class FactorGroup(Enum):
    Enum.Indices = "1-Risk Indices"
    Enum.Industries =  "2-Industries"
    Enum.Market = "5-Market"

@unique
class Factor(Enum):
    Enum.AERODEF = 'AERODEF'
    Enum.AIRLINE = 'AIRLINE'
    Enum.AUTO = 'AUTO'
    Enum.BANKS = 'BANKS'
    Enum.BETA = 'BETA'
    Enum.BEV = 'BEV'
    Enum.BLDPROD = 'BLDPROD'
    Enum.BTOP = 'BTOP'
    Enum.CHEM = 'CHEM'
    Enum.CNSTENG = 'CNSTENG'
    Enum.COMSERV = 'COMSERV'
    Enum.CONMAT = 'CONMAT'
    Enum.CONSSERV = 'CONSSERV'
    Enum.COUNTRY = 'COUNTRY'
    Enum.DVFININS = 'DVFININS'
    Enum.EARNYILD = 'EARNYILD'
    Enum.ELECEQP = 'ELECEQP'
    Enum.ENERGY = 'ENERGY'
    Enum.FOODPROD = 'FOODPROD'
    Enum.GROWTH = 'GROWTH'
    Enum.HDWRSEMI = 'HDWRSEMI'
    Enum.HEALTH = 'HEALTH'
    Enum.HOUSEDUR = 'HOUSEDUR'
    Enum.INDCONG = 'INDCONG'
    Enum.LEISLUX = 'LEISLUX'
    Enum.LEVERAGE = 'LEVERAGE'
    Enum.LIQUIDTY = 'LIQUIDTY'
    Enum.MACH = 'MACH'
    Enum.MARINE = 'MARINE'
    Enum.MATERIAL = 'MATERIAL'
    Enum.MEDIA = 'MEDIA'
    Enum.MOMENTUM = 'MOMENTUM'
    Enum.MTLMIN = 'MTLMIN'
    Enum.PERSPRD = 'PERSPRD'
    Enum.RDRLTRAN = 'RDRLTRAN'
    Enum.REALEST = 'REALEST'
    Enum.RESVOL = 'RESVOL'
    Enum.RETAIL = 'RETAIL'
    Enum.SIZE = 'SIZE'
    Enum.SIZENL = 'SIZENL'
    Enum.SOFTWARE = 'SOFTWARE'
    Enum.TRDDIST = 'TRDDIST'
    Enum.UTILITIE = 'UTILITIE'


@log
def get_barra_fac_info(factorgroup: FactorGroup = FactorGroup.Indices) -> DataFrame:
    """  """
    """因子序列，用以在矩阵乘法前统一因子顺序。也返回因子所属类别。
    修改历史：
        初稿 陈开琛 2018-03-30
        2018-04-26 已改用get_barra_expo返回值的index/col 作为统一。此函数仅供备用。
    :return:  43 * 2 DataFrame
        columns = ["FACTOR", "FACTORGROUP"]
        FACTORGROUP: ["1-Risk Indices": 10, "2-Industries": 32, "5-Market": 1]
    """

    if factorgroup == None:
        sql = " select substr(factor, 7, 20) factor, factorgroup" \
              " from xedm_trd.barra_factors_dat"
    else:
        sql = " select substr(factor, 7, 20) factor, factorgroup" \
              " from xedm_trd.barra_factors_dat" \
              " where factorgroup = '" + factorgroup + "'"
    fac_info = execute_sql(sql)
    fac_info.sort_values(by="FACTOR", ascending=True, inplace=True)
    return fac_info


@log
def get_barra_cov(beg_date: str, end_date: str,
                  factorgroup: FactorGroup = FactorGroup.Indices) -> DataFrame:
    """返回区间内每一日的因子协方差矩阵。
    修改历史：
        初稿 陈开琛 2018-03-30
    :param beg_date:
    :param end_date:
    :return: DataFrame。双重索引['T_DATE', 'FAC1']
    """

    if factorgroup != None :
        sql =   " select to_char(datadate,'yyyy-mm-dd') t_date, substr(fac1, 7, 20) fac1, substr(fac2, 7, 20) fac2, VARCOVAR"\
                " from "\
                " (select datadate, factor1 as fac1, factor2 as fac2, VARCOVAR"\
                " from"\
                " (select datadate, factor1, factor2, VARCOVAR"\
                "   from XEDM_TRD.BARRA_COVARIANCE"\
                "   where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd')"\
                " ) cov1"\
                " join "\
                " (select factor"\
                "   from XEDM_TRD.BARRA_FACTORS_DAT "\
                "   where FACTORGROUP='" + factorgroup + "'"\
                " ) group1"\
                " on group1.factor = cov1.factor1"\
                " join"\
                " (select factor"\
                "   from XEDM_TRD.BARRA_FACTORS_DAT "\
                "   where FACTORGROUP='" + factorgroup + "'"\
                " ) group2"\
                " on group2.factor = cov1.factor2"\
                " UNION"\
                " select datadate, FACTOR2 as fac1, FACTOR1 as fac2, VARCOVAR"\
                " from"\
                " (select datadate, factor1, factor2, VARCOVAR"\
                "   from XEDM_TRD.BARRA_COVARIANCE"\
                "   where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd')"\
                "     and FACTOR2 <> FACTOR1"\
                " ) cov2"\
                " join "\
                " (select factor"\
                "   from XEDM_TRD.BARRA_FACTORS_DAT "\
                "   where FACTORGROUP='" + factorgroup + "'"\
                " ) group1"\
                " on group1.factor = cov2.factor1"\
                " join"\
                " (select factor"\
                "   from XEDM_TRD.BARRA_FACTORS_DAT "\
                "   where FACTORGROUP='" + factorgroup + "'"\
                " ) group2"\
                " on group2.factor = cov2.factor2"\
                " )order by t_date, FAC1, FAC2"

    else:   # factorgroup == None, include all factors.
        sql =   "  select * from "\
                "  ("\
                "  select to_char(datadate,'yyyy-mm-dd') t_date, substr(factor1, 7, 20) fac1, substr(factor2, 7, 20) fac2, VARCOVAR"\
                "  from XEDM_TRD.BARRA_COVARIANCE"\
                "  where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd')"\
                "  UNION"\
                "  select to_char(datadate,'yyyy-mm-dd') t_date, substr(factor2, 7, 20) fac1 ,substr(factor1, 7, 20) fac2,VARCOVAR"\
                "  from XEDM_TRD.BARRA_COVARIANCE"\
                "  where FACTOR2 <> FACTOR1"\
                "   and DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd')"\
                "  )"\
                "  order by t_date, FAC1, FAC2"

    data_covariance = execute_sql(sql)
    data_covariance = data_covariance.pivot_table(index=['T_DATE', 'FAC1'], columns='FAC2', values='VARCOVAR')

    return data_covariance


@log
def get_barra_expo(beg_date: str, end_date: str, bench_mark: Index = Index.i000300,
        factorgroup: FactorGroup = FactorGroup.Indices, stock_list: list = None
        ) -> DataFrame:
    """ 返回区间内，每一日，每一个股，在每一因子上的暴露。
    修改历史：
        初稿 陈开琛 2018-03-30
    :param beg_date:
    :param end_date:
    :return: DataFrame。双重索引['T_DATE', 'ASSETID']
    """
    """
    注意：
        若某指数权重股范围发生变化，所有日期的返回值，都将全部包含在索引2ASSETID。但若权重股退市、再无暴露数据，则暴露为0.
        另见本函数末尾。

        例子：武钢于18年t日退市，则：

        >>>                BETA     GROWTH
        T_DATE ASSETID
        t-1     武钢       0.04       0.05
                权重股A    0.06       0.04
        t       武钢       0.0        0.0
                权重股A    -0.02      -0.03
        t+1     武钢       0.0        0.0
                权重股A    -0.03      -0.07


    """

    call_sql = time.time()

    if factorgroup != None:

        sql_factorgroup =   " join" \
                            " (select factor, factorgroup" \
                            "   from XEDM_TRD.BARRA_FACTORS_DAT" \
                            "   where FACTORGROUP='" + factorgroup + "'"\
                            " ) facgroup" \
                            " on expo_id.factor = facgroup.factor" \

        if bench_mark != None:  # only selected stock in bench

            union_line = ' '
            if stock_list != None:
                for index, stock in enumerate(stock_list):
                    union_line = union_line + "union (select '" + stock + "' i_code from dual) "

            sql =   " select to_char(datadate,'yyyy-mm-dd') t_date, ASSETID, substr(expo_id.factor, 7, 20) factor, exposure" \
                    " from" \
                    " (select datadate, ASSETID, factor, exposure" \
                    "   from" \
                    "   (select distinct datadate, substr(ASSETID,3,6) ASSETID, factor, exposure" \
                    "     from" \
                    "     (select datadate, barrid, factor, exposure" \
                    "       from xedm_trd.barra_Asset_Exposure" \
                    "       where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') "\
                    "     ) expoall" \
                    "     join" \
                    "     (select barrid, ASSETID" \
                    "       from XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID" \
                    "     ) id" \
                    "     on expoall.barrid = id.barrid" \
                    "   ) expo" \
                    "   join" \
                    "   (select distinct S_INFO_CODE" \
                    "       from V_INDEX_RTN_DETAIL " \
                    "       where wind_idx_code = '" + bench_mark + "' "\
                    "         and trade_date BETWEEN '" + beg_date + "' and '" + end_date + "' "\
                    +    union_line \
                    + "  ) bench"\
                    "    on ASSETID = S_INFO_CODE" \
                    " ) expo_id" \
                    + sql_factorgroup \
                    + " order by datadate, assetid, expo_id.factor"

        else:   # bench_mark == None, include all stocks.
            sql =   " select to_char(datadate,'yyyy-mm-dd') t_date, ASSETID, substr(expo_id.factor, 7, 20) factor, exposure" \
                    " from" \
                    " (select datadate, ASSETID, factor, exposure" \
                    "   from" \
                    "   (select distinct datadate, substr(ASSETID,3,6) ASSETID, factor, exposure" \
                    "     from" \
                    "     (select datadate, barrid, factor, exposure" \
                    "       from xedm_trd.barra_Asset_Exposure" \
                    "       where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') "\
                    "     ) expoall" \
                    "     join" \
                    "     (select barrid, ASSETID" \
                    "       from XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID" \
                    "     ) id" \
                    "     on expoall.barrid = id.barrid" \
                    "   ) expo" \
                    " ) expo_id" \
                    + sql_factorgroup \
                    + " order by datadate, assetid, expo_id.factor"

    else:   # factorgroup == None, include all factors.

        if bench_mark != None:

            union_line = ' '
            if stock_list != None:
                for index, stock in enumerate(stock_list):
                    union_line = union_line + "union (select '" + stock + "' i_code from dual) "

            sql =   " select to_char(datadate,'yyyy-mm-dd') t_date, ASSETID, substr(expo_id.factor, 7, 20) factor, exposure" \
                    " from" \
                    " (select datadate, ASSETID, factor, exposure" \
                    "   from" \
                    "   (select distinct datadate, substr(ASSETID,3,6) ASSETID, factor, exposure" \
                    "     from" \
                    "     (select datadate, barrid, factor, exposure" \
                    "       from xedm_trd.barra_Asset_Exposure" \
                    "       where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') "\
                    "     ) expoall" \
                    "     join" \
                    "     (select barrid, ASSETID" \
                    "       from XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID" \
                    "     ) id" \
                    "     on expoall.barrid = id.barrid" \
                    "   ) expo" \
                    "   join" \
                    "   (select distinct S_INFO_CODE" \
                    "       from V_INDEX_RTN_DETAIL " \
                    "       where wind_idx_code = '" + bench_mark + "' "\
                    "         and trade_date BETWEEN '" + beg_date + "' and '" + end_date + "' "\
                    +    union_line \
                    + "  ) bench"\
                    "    on ASSETID = S_INFO_CODE" \
                    " ) expo_id" \
                    " order by datadate, assetid, expo_id.factor"

        else: # include all factor, all stock.
            sql =   "select to_char(datadate,'yyyy-mm-dd') t_date, substr(ASSETID,3,6) ASSETID, substr(factor, 7, 20) factor, exposure" \
                    " from" \
                    " (select * from xedm_trd.barra_Asset_Exposure" \
                    " where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') ) expo" \
                    " join" \
                    " XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID id" \
                    " on expo.barrid = Id.Barrid" \
                    " order by T_DATE, ASSETID "

    print(sql)

    expo = execute_sql(sql)
    # return_sql = time.time()

    # expo = expo.sort_values(['T_DATE', 'ASSETID', 'FACTOR'])
    expo = expo.pivot_table(
        index = ['T_DATE', 'ASSETID'],columns = 'FACTOR',
        values='EXPOSURE').fillna(0)

    """
    从 V_INDEX_RTN_DETAIL 表中，取出整个区间所有股票的代码，并fillna，意味着：
        若区间内，有30只权重股被替换，则股票代码总数为330（每一日的行数为330）。
        若某些股票无暴露数据（如武钢退市），则会补充为0.
    在其他表中，沿用这一做法，如bench_wgt，以确保数据缺失的股票上没有权重、收益。
    """
    # return_expo_data = time.time()
    # print("SQL execution:", return_sql - call_sql)
    # print("Loading data:", return_expo_data - return_sql, "\nExiting get_barra_expo")
    # list_factor = list(get_barra_fac_info().loc[:, 'FACTOR'])
    # expo = expo.loc[:, list_factor]   # 统一因子顺序。
    return expo

@log
def get_selected_stock_rtn(beg_date: str, end_date: str, period: Period = Period.MONTH,
        stock_list: list = None, bench_mark: Index = Index.i000300
                           ) -> DataFrame:
    """返回区间内，每一日，hs300权重股的收益率
    修改历史：
        初稿 陈开琛 2018-04-10
        可选计算频率，选股，或选指数权重股。 陈开琛 2018-04-17
    :param beg_date:
    :param end_date:
    :return: DataFrame。双重索引['T_DATE', 'ASSETID']
    """
    """
    注意：
        若某指数权重股范围发生变化，新权重股在加入前指数，不会显示为一行。
        这与get_barra_expo的逻辑完全不同。
        主要目的，是在回测时，用这一函数，规范每一个交易日的选股范围。

        例子：武钢于18年t日退市，则：

        >>>                RTN
        T_DATE ASSETID
        t-1     武钢       0.04
                权重股A    0.06
        t(退市) 权重股A    0.02
        t+1     权重股A    -0.05
    """

    if  bench_mark != None:

        union_line =    " "
        print(stock_list)
        if stock_list != None:
            if len(stock_list) != 0:
                union_line = union_line +   " union ( select trade_dt, i_code" \
                                            " from" \
                                            " (select distinct trade_dt" \
                                            "   from V_INDEX_RTN_DETAIL" \
                                            "   where wind_idx_code = '000905.SH'       " \
                                            "     and trade_date BETWEEN '2017-04-18' and '2017-07-17'" \
                                            " ) dates" \
                                            " cross join" \
                                            " ("
                for i, stock in enumerate(stock_list):
                    if i != len(stock_list)-1:
                        union_line = union_line + " (select '" + stock + "' i_code from dual) union "
                    else:
                        union_line = union_line + " (select '" + stock + "' i_code from dual)) self_stock )  "

        sql =   " select to_char(to_date(datadate, 'YYYYMMDD'), 'YYYY-MM-DD') T_DATE, ASSETID, dlyreturn/100 as RTN"\
                " from"\
                " (select distinct S_INFO_CODE i_code, trade_dt" \
                "    from V_INDEX_RTN_DETAIL " \
                "    where wind_idx_code = '" + bench_mark + "' " \
                "      and trade_date BETWEEN '" + beg_date + "' and '" + end_date + "'  " \
                + union_line \
                + " ) bench"\
                " left join"\
                " (select datadate, assetid, dlyreturn"\
                "   from"\
                "   (select to_char(DATADATE, 'yyyymmdd') datadate, barrid, dlyreturn"\
                "     from xedm_trd.barra_Daily_Asset_Price "\
                "     where DATADATE between to_date('" + beg_date + "', 'yyyy-mm-dd') and to_date('" + end_date + "', 'yyyy-mm-dd')"\
                "   ) atdlyret"\
                "   join"\
                "   ("\
                "     select distinct id.BARRID, substr(id.ASSETID,3,6) ASSETID"\
                "     from XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID id"\
                "     left join XEDM_TRD.barra_chn_asset_identity a"\
                "     on id.BARRID = a.BARRID where a.instrument = 'STOCK'"\
                "   ) id"\
                "   on atdlyret.barrid = id.barrid"\
                " ) barra"\
                " on bench.i_code = barra.assetid"\
                " and bench.trade_dt = barra.datadate"\
                " order by t_date, assetid"

    else:   # bench_mark == None

        if stock_list != None:
            or_line = " ( "
            for i, stock in enumerate(stock_list):
                if i == len(stock_list)-1:
                    or_line = or_line + " S_INFO_WINDCODE like '" + stock + "%') "
                else:
                    or_line = or_line + " S_INFO_WINDCODE like '" + stock + "%' OR "

            sql =   " select to_char(to_date(trade_dt,'YYYYMMDD'),'YYYY-MM-DD') T_DATE, " \
                    "   substr(S_INFO_WINDCODE, 1, 6) ASSETID, S_Dq_Pctchange/100 RTN" \
                    " from xedm_trd.AShareEODPrices" \
                    " where trade_dt between replace('" + beg_date + "', '-', '') and replace('" + end_date + "', '-', '')" \
                    "   and " + or_line

        else:   # bench_mark == None, stock_list == None, include all stocks.
            sql =   " select to_char(to_date(trade_dt,'YYYYMMDD'),'YYYY-MM-DD') T_DATE, " \
                    "   substr(S_INFO_WINDCODE, 1, 6) ASSETID, S_Dq_Pctchange/100 RTN" \
                    " from xedm_trd.AShareEODPrices" \
                    " where trade_dt between replace('" + beg_date + "', '-', '') and replace('" + end_date + "', '-', '')"

    stock_rtn = execute_sql(sql)

    if period == 'WEEK':
        stock_rtn['T_DATETIME'] = pd.to_datetime(stock_rtn["T_DATE"])
        stock_rtn['YEAR'] = stock_rtn["T_DATETIME"].apply(lambda x: x.isocalendar()[0])
        stock_rtn['WEEK'] = stock_rtn["T_DATETIME"].apply(lambda x: x.isocalendar()[1])
        stock_rtn = pd.DataFrame({
            "T_DATE": stock_rtn.groupby(['YEAR', 'WEEK', 'ASSETID'])["T_DATETIME"].agg(lambda x: max(x).strftime("%Y-%m-%d")),
            "RTN": stock_rtn.groupby(['YEAR', 'WEEK', 'ASSETID'])["DLY_RTN"].agg(lambda x: (x.fillna(0) + 1).prod() - 1)
        }).reset_index(inplace=False).set_index(['T_DATE', 'ASSETID'],inplace=False).sort_index(ascending = True)

    elif period == 'MONTH':
        stock_rtn['T_DATETIME'] = pd.to_datetime(stock_rtn["T_DATE"])
        stock_rtn['YEAR'] = stock_rtn["T_DATETIME"].apply(lambda x: x.isocalendar()[0])
        stock_rtn['MONTH'] = stock_rtn["T_DATE"].apply(lambda x: x[0:7])
        stock_rtn = pd.DataFrame({
            "T_DATE": stock_rtn.groupby(['YEAR', 'MONTH', 'ASSETID'])["T_DATETIME"].agg(lambda x: max(x).strftime("%Y-%m-%d")),
            "RTN": stock_rtn.groupby(['YEAR', 'MONTH', 'ASSETID'])["DLY_RTN"].agg(lambda x: (x.fillna(0) + 1).prod() - 1)
        }).reset_index(inplace=False).set_index(['T_DATE', 'ASSETID'],inplace=False).sort_index(ascending = True)

    else:
        stock_rtn = stock_rtn.pivot_table(values = "RTN", index = ['T_DATE', 'ASSETID']).sort_index(ascending = True)

    print(sql)

    return stock_rtn

@log
def get_barra_fac_rtn(beg_date: str, end_date: str,
        factorgroup: FactorGroup = FactorGroup.Indices ) -> DataFrame:
    """ 返回区间内，每一日，每一因子的因子回报。
    修改历史：
        初稿 陈开琛 2018-03-30
        加入factorgroup 陈开琛 18-04-13
    :param beg_date:
    :param end_date:
    :param factorgroup:
    :return: DataFrame。
    """
    sql =   " select to_char(datadate,'yyyy-mm-dd') t_date, substr(factor, 7, 20) factor, dlyreturn" \
            " from xedm_trd.barra_dlyfacret" \
            " where datadate between  to_date('" + beg_date + "', 'yyyy-mm-dd') and to_date('" + end_date + "', 'yyyy-mm-dd')"\
            "    and factor in (select factor from XEDM_TRD.BARRA_FACTORS_DAT" \
            "                   where FACTORGROUP='" + factorgroup + "')" \
            " order by t_date, factor"\

    print(sql)
    fac_rtn = execute_sql(sql)

    if fac_rtn is not None and fac_rtn.size != 0:
        fac_rtn = fac_rtn.pivot_table(index = "T_DATE", columns = "FACTOR", values = "DLYRETURN").fillna(0)
        return fac_rtn

    raise Exception("SQL Request return nothing.")

@log
def get_bench_weight(beg_date: str, end_date: str, bench_mark: Index = Index.i000300)-> DataFrame:
    """
    获取指数权重股的权重。
    修改历史：
    初稿 陈开琛  2018-04-13
    :param beg_date:
    :param end_date:
    :param bench_mark:
    :return:
    """
    """
    注意：
        权重股变化时，bench_wgt的列包含所有权重股。

    例子：

        >>>ASSETID 权重股A    权重股B    武钢
        T_DATE
        t-1     0.03        0.04        0.10
        t(退市) 0.04        0.06        0
        t+1     0.05        0.02        0
    """

    sql =   " select trade_date T_DATE, S_INFO_CODE ASSETID, i_weight" \
            " from V_INDEX_RTN_DETAIL " \
            " where wind_idx_code = '" + bench_mark + "' " \
            "   and trade_date between '" + beg_date + "' and '" + end_date + "'" \
            " order by Trade_Date, S_Info_Code"

    bench_wgt = execute_sql(sql)
    bench_wgt = bench_wgt.pivot_table(
        index = "T_DATE", columns= "ASSETID", values="I_WEIGHT"
    ).fillna(0)
    """
    必须pivot_table.fillna。（只针对取多日bench_wgt的情况）
    以确保权重股变化时，bench_wgt的列包含所有权重股
        （如，6月12日后，hs300更换了30只成分股，此时应有330 个column）
    并且，没有数据的列，权重为0.
    """
    return bench_wgt

@log
def get_barra_spec_risk(beg_date: str, end_date: str, stock_list: list = None, bench_mark: Index = Index.i000300) -> DataFrame:
    """
    获取个股特异风险。

    修改历史：
    初稿 陈开琛  2018-03-30
    增强 陈开琛  2018-04-23
    :param beg_date:
    :param end_date:
    :param stock_list:
    :param bench_mark:
    :return:
    """

    if  bench_mark != None:
        union_line = ' '
        if stock_list != None:
            for stock in stock_list:
                union_line = union_line + " union (select '" + stock + "' S_INFO_CODE from dual) "

        sql =   " select to_char(datadate,'yyyy-mm-dd') T_DATE, substr(ASSETID,3,6) ASSETID, specrisk SPEC_RISK" \
                " from " \
                " (select distinct Risk.datadate, id.ASSETID, Specrisk " \
                "   from" \
                "   (select datadate, barrid, Specrisk from xedm_trd.barra_asset_data" \
                "     where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') " \
                "   ) Risk" \
                "   join" \
                "   (select * " \
                "     from XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID" \
                "   ) id" \
                "   on Risk.barrid = Id.Barrid" \
                " ) riskid" \
                " join" \
                " (select distinct S_INFO_CODE       " \
                "     from V_INDEX_RTN_DETAIL" \
                "     where wind_idx_code = '" + bench_mark + "'          " \
                "       and trade_date BETWEEN '" + beg_date + "' and '" + end_date + "'  " \
                + union_line \
                + " ) bench" \
                " on substr(riskid.ASSETID,3,6) = S_INFO_CODE" \
                " order by T_DATE, ASSETID"

    else:
        sql =   " select to_char(datadate,'yyyy-mm-dd') T_DATE, substr(ASSETID,3,6) ASSETID, specrisk SPEC_RISK" \
                " from " \
                " (select * from xedm_trd.barra_asset_data" \
                " where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') ) SpecRisk" \
                " join" \
                " XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID id" \
                " on SpecRisk.barrid = Id.Barrid"

    spec_risk = execute_sql(sql)
    spec_risk = spec_risk.pivot_table(values="SPEC_RISK", index="T_DATE", columns="ASSETID").fillna(0)

    return spec_risk


@log
def get_barra_spec_rtn(beg_date: str, end_date: str) -> DataFrame:
    """
    获取个股特异收益。

    修改历史：
    初稿 陈开琛  2018-03-30
    :param beg_date:
    :param end_date:
    :return:
    """
    sql = " select to_char(datadate,'yyyy-mm-dd') t_date, substr(ASSETID,3,6) ASSETID, specificreturn" \
          " from " \
          " (select * from xedm_trd.barra_Asset_DlySpecRet" \
          " where DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd') ) SpecRtn" \
          " join" \
          " XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID id" \
          " on SpecRtn.barrid = Id.Barrid"
    spec_rtn = execute_sql(sql)
    spec_rtn = spec_rtn.pivot_table(values="SPECIFICRETURN", index="T_DATE", columns="ASSETID")
    return spec_rtn


@log
def get_barra_expo_hs300(beg_date: str, end_date: str) -> DataFrame:
    """
    180413 已废弃。要能提供factor_group、benchmark范围的限制，非常耗时，可以直接用
            已经整理过范围的exposure数据直接与bench_wgt作矩阵乘法。
        改为get_bench_weight()
    :param beg_date:
    :param end_date:
    :return:
    """
    sql = " select hs.t_date, substr(ba.FACTOR, 7, 20) as FACTOR, sum(hs.i_weight / 100 * ba.EXPOSURE) EXPOSURE"\
            " from "\
            "   (select substr(S_CON_WINDCODE, 1, 6) i_code, to_char(to_date(trade_dt, 'YYYYMMDD'), 'YYYY-MM-DD') t_date, i_weight"\
            "     from wdifdata.AIndexHS300Weight"\
            "     where TRADE_DT between replace('" + beg_date + "', '-', '') and replace('" + end_date + "', '-', '')"\
            " ) hs "\
            " left join "\
            "   (select id.BARRID, id.ASSETID, e.FACTOR, e.EXPOSURE, to_char(e.DATADATE, 'yyyy-mm-dd') t_date"\
            "     from "\
            "       (select distinct id.BARRID, substr(id.ASSETID, 3, 6) ASSETID"\
            "         from XEDM_TRD.BARRA_CHN_LOCALID_ASSET_ID id"\
            "         left join XEDM_TRD.barra_chn_asset_identity a"\
            "           on id.BARRID = a.BARRID"\
            "         where a.instrument = 'STOCK'"\
            "     ) id "\
            "     left join xedm_trd.barra_asset_exposure e"\
            "     on id.BARRID = e.BARRID"\
            "     where e.DATADATE between to_date('" + beg_date + "','yyyy-mm-dd') and to_date('" + end_date + "','yyyy-mm-dd')"\
            " ) ba "\
            " on hs.i_code = ba.ASSETID"\
            "   and hs.t_date = ba.t_date"\
            " group by hs.t_date, ba.FACTOR"\
            " order by T_DATE, FACTOR"
    expo_hs300 = execute_sql(sql)
    expo_hs300 = expo_hs300.pivot_table(index = 'T_DATE', columns = 'FACTOR', values='EXPOSURE')
    expo_hs300 = expo_hs300.fillna(0)

    return expo_hs300





