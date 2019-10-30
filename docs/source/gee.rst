.. currentmodule:: statsmodels.genmod.generalized_estimating_equations

.. _gee:

广义估计方程
================================

广义估计方程是针对观测值的面板、与其中一个聚类相关但与各个聚类不相关和非独立时，
从广义线性模型发展而来，它具有相同的广义线性模型(`GLM`) 的 one-parameter 指数族

有关命令和参数，请看 `Module Reference`_ 

例子
--------

在聚类时使用癫痫发作数据来说明具有可交换相关性的泊松分布回归

.. ipython:: python

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    data = sm.datasets.get_rdataset('epil', package='MASS').data

    fam = sm.families.Poisson()
    ind = sm.cov_struct.Exchangeable()
    mod = smf.gee("y ~ age + trt + base", "subject", data,
                  cov_struct=ind, family=fam)
    res = mod.fit()
    print(res.summary())


在维基百科可以找到几个使用GEE模型的示例：
`GEE的维基笔记 <https://github.com/statsmodels/statsmodels/wiki/Examples#generalized-estimating-equations-gee>`_


参考文献
^^^^^^^^^^

* KY Liang 和 S Zeger. “使用广义线性模型进行纵向数据分析”. Biometrika (1986) 73 (1): 13-22.
* S Zeger 和 KY Liang. “用于离散和连续结果的纵向数据分析”. Biometrics Vol. 42, No. 1 (Mar., 1986),
  pp. 121-130
* A Rotnitzky 和 NP Jewell (1990). “聚类相关数据的半参数广义线性模型中回归参数的假设检验”, Biometrika, 77, 485-497.
* Xu Guo 和 Wei Pan (2002). “ GEE中分数测试的小样本性能”.
  http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
* LA Mancl LA, TA DeRouen (2001). 具有改进的小样本属性的GEE协方差估计器.  生物识别. 2001 Mar;57(1):126-34.


模块参考
----------------

.. module:: statsmodels.genmod.generalized_estimating_equations
   :synopsis: Generalized estimating equations

模型类
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GEE
   NominalGEE
   OrdinalGEE

.. module:: statsmodels.genmod.qif
   :synopsis: Quadratic inference functions

.. currentmodule:: statsmodels.genmod.qif

.. autosummary::
   :toctree: generated/

   QIF

结果类
^^^^^^^^^^^^^^^

.. currentmodule:: statsmodels.genmod.generalized_estimating_equations

.. autosummary::
   :toctree: generated/

   GEEResults
   GEEMargins

.. currentmodule:: statsmodels.genmod.qif

.. autosummary::
   :toctree: generated/

   QIFResults

依赖结构
^^^^^^^^^^^^^^^^^^^^^

当前实现的依赖结构是

.. module:: statsmodels.genmod.cov_struct
   :synopsis: Covariance structures for Generalized Estimating Equations (GEE)

.. currentmodule:: statsmodels.genmod.cov_struct

.. autosummary::
   :toctree: generated/

   CovStruct
   Autoregressive
   Exchangeable
   GlobalOddsRatio
   Independence
   Nested


家族模型
^^^^^^^^

与 GLM 相同，目前已实现

.. module:: statsmodels.genmod.families.family
   :synopsis: Generalized Linear Model (GLM) families

.. currentmodule:: statsmodels.genmod.families.family

.. autosummary::
   :toctree: generated/

   Family
   Binomial
   Gamma
   Gaussian
   InverseGaussian
   NegativeBinomial
   Poisson
   Tweedie


Link 函数
^^^^^^^^^^^^^^

GEE的Link 函数与GLM相同, 下面是目前已实现的函数. 并不是每个家族模型都可以使用 link 函数。
可以使用 link 函数的如下表.

::

    >>> sm.families.family.<familyname>.links

.. currentmodule:: statsmodels.genmod.families.links

.. autosummary::
   :toctree: generated/

   Link

   CDFLink
   CLogLog
   Log
   Logit
   NegativeBinomial
   Power
   cauchy
   cloglog
   identity
   inverse_power
   inverse_squared
   log
   logit
   nbinom
   probit
