import unittest
from b_model.port_opt.port_opt import *


def get_bench_stock(t_date: str, bench: Index)->DataFrame:
    sql =   " select s_info_code" \
            " from V_INDEX_RTN_DETAIL " \
            " where trade_date BETWEEN '" + t_date + "' and '" + t_date + "' " \
            "   and wind_idx_code = '" + bench + "'"
    stocks = execute_sql(sql)
    return stocks

class Test_Min_Var(unittest.TestCase):
    """
    有无制定基准
    有无新增个股
    有无制定中性因子
    有无设定持仓上限
    中性种类
    """
    def test_bench_hs300(self):
        t_date, bench = "2017-04-18", Index.i000300

        opt = Neutral_Port_Opt(t_date, bench)
        wgt = opt.get_min_var_port(neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 0)
        self.assertEqual(len(dif2), 0)

    def test_bench_000905(self):
        t_date, bench = "2017-04-18", Index.i000905

        opt = Neutral_Port_Opt(t_date, bench)
        wgt = opt.get_min_var_port(neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 0)
        self.assertEqual(len(dif2), 0)

    def test_selected_off_bench_stock(self):
        t_date, bench = "2017-04-18", Index.i000300
        stock_selected = pd.DataFrame({
            "ASSETID": ['000001', '000006'],
            "exp_rtn": [None, None]
        })

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_stock_exp_rtn(stock_selected)
        wgt = opt.get_min_var_port(neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(dif1[0], '000006')
        self.assertEqual(len(dif2), 0)

    def test_fac_neutralized(self):
        t_date, bench, neu_fac = "2017-04-18", Index.i000300, Factor.SIZE
        print(neu_fac)

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_min_var_port(neu_type=Neutral_Type.Zero)

        port_expo = np.dot( get_barra_expo(t_date, t_date, bench)[neu_fac].loc[t_date].T, wgt )
        print(port_expo)
        self.assertAlmostEqual(port_expo, 0.0, 2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_neu_type(self):
        t_date, bench, neu_fac = "2017-04-18", Index.i000300, Factor.SIZE
        print(neu_fac)

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_min_var_port(neu_type=Neutral_Type.Benchmark)

        expo = get_barra_expo(t_date, t_date, bench)[neu_fac].loc[t_date].T
        port_expo = np.dot( expo, wgt )
        bench_wgt = get_bench_weight(t_date, t_date, bench).loc[t_date, expo.index.tolist()]
        bench_expo = np.dot( expo, bench_wgt )
        self.assertAlmostEqual(port_expo, bench_expo, 2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_max_weight(self):
        t_date, bench, neu_fac, max_wgt = "2017-04-18", Index.i000300, Factor.SIZE, 0.06

        opt = Neutral_Port_Opt(t_date, bench, max_weight=max_wgt)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_min_var_port(neu_type=Neutral_Type.Zero)

        self.assertTrue(np.allclose(wgt.tolist(), np.full((len(wgt), 1), max_wgt/2), atol = max_wgt/2))
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)


class Test_Max_Rtn(unittest.TestCase):
    """
    有无主观设定预期收益
    有无制定基准
    有无新增个股
    有无制定中性因子
    有无设定持仓上限
    中性种类
    """

    def test_without_exp_rtn(self):
        t_date, bench = "2017-04-18", Index.i000300

        opt = Neutral_Port_Opt(t_date, bench)
        # opt.set_stock_exp_rtn(stock_selected)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 0)
        self.assertEqual(len(dif2), 0)


    def test_bench_hs300(self):
        t_date, bench = "2017-04-18", Index.i000300
        stock_selected = pd.DataFrame({
            "ASSETID": ['000001', '000008'],
            "exp_rtn": [0.16, 0.20]
        })

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_stock_exp_rtn(stock_selected)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 0)
        self.assertEqual(len(dif2), 0)

    def test_bench_000905(self):
        t_date, bench = "2017-04-18", Index.i000905
        stock_selected = pd.DataFrame({
            "ASSETID": ['000001', '000008'],
            "exp_rtn": [0.16, 0.20]
        })

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_stock_exp_rtn(stock_selected)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        print(dif1)
        print(dif2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 2)
        self.assertEqual(len(dif2), 0)

    def test_selected_off_bench_stock(self):
        t_date, bench = "2017-04-18", Index.i000300
        neu_fac = Factor.SIZE
        stock_selected = pd.DataFrame({
            "ASSETID": ['000006'],
            "exp_rtn": [0.20]
        })

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_stock_exp_rtn(stock_selected)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        port_expo = np.dot( get_barra_expo(t_date, t_date, bench, stock_list=['000006'])[neu_fac].loc[t_date].T, wgt )
        print(port_expo)
        self.assertAlmostEqual(port_expo, 0.0, 2)

        self.assertEqual(dif1[0], '000006')
        self.assertEqual(len(dif2), 0)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_fac_neutralized(self):
        t_date, bench, neu_fac = "2017-04-18", Index.i000300, Factor.SIZE
        print(neu_fac)

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)

        port_expo = np.dot( get_barra_expo(t_date, t_date, bench)[neu_fac].loc[t_date].T, wgt )
        print(port_expo)
        self.assertAlmostEqual(port_expo, 0.0, 2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_neu_type(self):
        t_date, bench, neu_fac = "2017-04-18", Index.i000300, Factor.SIZE
        print(neu_fac)

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Benchmark)

        expo = get_barra_expo(t_date, t_date, bench)[neu_fac].loc[t_date].T
        port_expo = np.dot( expo, wgt )
        bench_wgt = get_bench_weight(t_date, t_date, bench).loc[t_date, expo.index.tolist()]
        bench_expo = np.dot( expo, bench_wgt )
        self.assertAlmostEqual(port_expo, bench_expo, 2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_max_weight(self):
        t_date, bench, neu_fac, max_wgt = "2017-04-18", Index.i000300, Factor.SIZE, 0.06

        opt = Neutral_Port_Opt(t_date, bench, max_weight=max_wgt)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_max_rtn_port(3, neu_type=Neutral_Type.Zero)

        self.assertTrue(np.allclose(wgt.tolist(), np.full((len(wgt), 1), max_wgt/2), atol = max_wgt/2))
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)


class Test_Min_Te(unittest.TestCase):
    """
    有无制定基准
    有无新增个股
    有无制定中性因子
    有无设定持仓上限
    中性种类
    """
    def test_bench_hs300(self):
        t_date, bench = "2017-04-18", Index.i000300

        opt = Neutral_Port_Opt(t_date, bench)
        wgt = opt.get_min_te_port(neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 0)
        self.assertEqual(len(dif2), 0)

    def test_bench_000905(self):
        t_date, bench = "2017-04-18", Index.i000905

        opt = Neutral_Port_Opt(t_date, bench)
        wgt = opt.get_min_te_port(neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(len(dif1), 0)
        self.assertEqual(len(dif2), 0)

    def test_selected_off_bench_stock(self):
        t_date, bench = "2017-04-18", Index.i000300
        stock_selected = pd.DataFrame({
            "ASSETID": ['000001', '000006'],
            "exp_rtn": [None, None]
        })

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_stock_exp_rtn(stock_selected)
        wgt = opt.get_min_te_port(neu_type=Neutral_Type.Zero)

        wgt_index = wgt.index
        check_index = get_bench_stock(t_date, bench).set_index("S_INFO_CODE").index
        dif1 = wgt_index.difference(check_index).tolist()
        dif2 = check_index.difference(wgt_index).tolist()

        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
        self.assertEqual(dif1[0], '000006')
        self.assertEqual(len(dif2), 0)

    def test_fac_neutralized(self):
        t_date, bench, neu_fac = "2017-04-18", Index.i000300, Factor.SIZE
        print(neu_fac)

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_min_te_port(neu_type=Neutral_Type.Zero)

        port_expo = np.dot( get_barra_expo(t_date, t_date, bench)[neu_fac].loc[t_date].T, wgt )
        print(port_expo)
        self.assertAlmostEqual(port_expo, 0.0, 2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_neu_type(self):
        t_date, bench, neu_fac = "2017-04-18", Index.i000300, Factor.SIZE
        print(neu_fac)

        opt = Neutral_Port_Opt(t_date, bench)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_min_te_port(neu_type=Neutral_Type.Benchmark)

        expo = get_barra_expo(t_date, t_date, bench)[neu_fac].loc[t_date].T
        port_expo = np.dot( expo, wgt )
        bench_wgt = get_bench_weight(t_date, t_date, bench).loc[t_date, expo.index.tolist()]
        bench_expo = np.dot( expo, bench_wgt )
        self.assertAlmostEqual(port_expo, bench_expo, 2)
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)

    def test_max_weight(self):
        t_date, bench, neu_fac, max_wgt = "2017-04-18", Index.i000300, Factor.SIZE, 0.06

        opt = Neutral_Port_Opt(t_date, bench, max_weight=max_wgt)
        opt.set_neutral_fac(neu_fac)
        wgt = opt.get_min_te_port(neu_type=Neutral_Type.Zero)

        self.assertTrue(np.allclose(wgt.tolist(), np.full((len(wgt), 1), max_wgt/2), atol = max_wgt/2))
        self.assertAlmostEqual(wgt.sum(), 1.0, 4)
