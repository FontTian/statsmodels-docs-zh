.. currentmodule:: statsmodels.stats.anova

.. _anova:

方差分析
=====

方差分析模型包含线性最小二乘模型的方差线性的方差分析, 和 AnovaRM 用于重复测量的方差分析, 
以及用于平衡数据的方差分析

例子
--------

.. ipython:: python

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    moore = sm.datasets.get_rdataset("Moore", "carData",
                                     cache=True) # load data
    data = moore.data
    data = data.rename(columns={"partner.status":
                                "partner_status"}) # make name pythonic
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                    data=data).fit()

    table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame
    print(table)

更多方差线性的方差分析示例:

*  `ANOVA <examples/notebooks/generated/interactions_anova.html>`__

模块参考
----------------

.. module:: statsmodels.stats.anova
   :synopsis: Analysis of Variance

.. autosummary::
   :toctree: generated/

   anova_lm
   AnovaRM
