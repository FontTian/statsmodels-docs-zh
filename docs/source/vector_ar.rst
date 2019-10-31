:orphan:

.. module:: statsmodels.tsa.vector_ar.var_model
   :synopsis: Vector autoregressions

.. currentmodule:: statsmodels.tsa.vector_ar.var_model

.. _var:

向量自回归 :mod:`tsa.vector_ar`
===========================================

:mod:`statsmodels.tsa.vector_ar` 包含的方法可用于同时使用
:ref:`Vector Autoregressions (VAR) <var>` 和
:ref:`Vector Error Correction Models (VECM) <vecm>`. 进行建模和分析多个时间序列

.. _var_process:

VAR(p) 过程
----------------

我们感兴趣的是对 :math:`T \times K` 维的多元时间序列
:math:`Y` 进行建模， 其中 :math:`T` 表示 and :math:`K` 表示变量输. 
估计时间序列与其滞后值之间的关系的一种方法是向量自回归过程： *vector autoregression process*:

.. math::

   Y_t = \nu + A_1 Y_{t-1} + \ldots + A_p Y_{t-p} + u_t

   u_t \sim {\sf Normal}(0, \Sigma_u)

其中 :math:`A_i` 是一个 :math:`K \times K` 维的系数矩阵。

我们在很大程度上遵循了 `Lutkepohl (2005)
<https://www.springer.com/gb/book/9783540401728>`__,
的方法和符号，在此不再赘述。

模型拟合
~~~~~~~~~~~~~

.. 注意::

    以下引用的类可通过
    :mod:`statsmodels.tsa.api` 模块进行访问。.

要估算VAR模型，必须首先使用同构或结构化的 `ndarray` 创建模型.
 当使用结构化或记录 array 时，该类将使用传递的变量名称。
 否则，可以直接的的传递它们：
explicitly:

.. ipython:: python
   :suppress:

   import pandas as pd
   pd.options.display.max_rows = 10
   import matplotlib
   import matplotlib.pyplot as plt
   matplotlib.style.use('ggplot')

.. ipython:: python
   :okwarning:

   # 示例数据
   import numpy as np
   import pandas
   import statsmodels.api as sm
   from statsmodels.tsa.api import VAR
   mdata = sm.datasets.macrodata.load_pandas().data

   # 将时间作为 index
   dates = mdata[['year', 'quarter']].astype(int).astype(str)
   quarterly = dates["year"] + "Q" + dates["quarter"]
   from statsmodels.tsa.base.datetools import dates_from_str
   quarterly = dates_from_str(quarterly)

   mdata = mdata[['realgdp','realcons','realinv']]
   mdata.index = pandas.DatetimeIndex(quarterly)
   data = np.log(mdata).diff().dropna()

   # 创建 VAR 模型
   model = VAR(data)

.. 注意::

    :class:`VAR` 模型类假定传递时间序列是平稳的. 非平稳或趋势数据通常可以
    通过一阶微分或其他方法转换为平稳数据. 对于非平稳时间序列的直接分析，
    标准的稳定 VAR(p) 模型不合适

在实际进行估算时，请使用所需的滞后顺序调用 `fit` 方法。或者，您可以让模型
根据标准信息条件选择滞后顺序 (请参见下文):

.. ipython:: python
   :okwarning:

   results = model.fit(2)
   results.summary()

有几种使用 `matplotlib` 可视化数据的方法

绘制输入的时间序列:

.. ipython:: python
   :okwarning:

   @savefig var_plot_input.png
   results.plot()


绘制时间序列自相关函数:

.. ipython:: python

   @savefig var_plot_acorr.png
   results.plot_acorr()


滞后顺序选择
~~~~~~~~~~~~~~~~~~~

滞后顺序的选择可能是一个难题。标准分析采用可能性测试或基于信息准则的顺序选择。
我们已经实现了后者，可以通过 :class:`VAR` 类进行访问:

.. ipython:: python

   model.select_order(15)

调用 `fit` 函数时, 可以传递最大滞后数和用于顺序选择的顺序标准：

.. ipython:: python

   results = model.fit(maxlags=15, ic='aic')

预测
~~~~~~~~~~~

就均方误差而言，线性预测器是最优的 h-step 提前预测:

.. math::

   y_t(h) = \nu + A_1 y_t(h − 1) + \cdots + A_p y_t(h − p)

我们可以使用预测功能来生成此预测。请注意，我们必须为预测指定e "initial value(初始值)" :

.. ipython:: python

   lag_order = results.k_ar
   results.forecast(data.values[-lag_order:], 5)

 `forecast_interval` 函数将产生上述预测以及渐近标准误差。可以使用 `plot_forecast`
函数将其可视化:

.. ipython:: python

   @savefig var_forecast.png
   results.plot_forecast(10)

Class 参考
~~~~~~~~~~~~~~~

.. module:: statsmodels.tsa.vector_ar
   :synopsis: Vector autoregressions and related tools

.. currentmodule:: statsmodels.tsa.vector_ar.var_model


.. autosummary::
   :toctree: generated/

   VAR
   VARProcess
   VARResults


估计后分析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

向量自回归过程可以使用几种过程属性和估计后的其他结果。

.. currentmodule:: statsmodels.tsa.vector_ar.var_model
.. autosummary::
   :toctree: generated/

   LagOrderResults

.. currentmodule:: statsmodels.tsa.vector_ar.hypothesis_test_results
.. autosummary::
   :toctree: generated/

   HypothesisTestResults
   NormalityTestResults
   WhitenessTestResults


脉冲响应分析
-------------------------

*Impulse responses* 在计量经济学研究中很重要：它们是对变量之一中单位脉冲的估计响应。
实际上，它们是使用 VAR(p) 过程的 MA(:math:`\infty`) 表示来计算的:

.. math::

    Y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

我们可以通过在 `VARResults` 对象上调用 `irf` 函数来进行脉冲响应分析:

.. ipython:: python
   :okwarning:

   irf = results.irf(10)

这些可以使用绘图函数以正交或非正交形式可视化。默认情况下，渐近标准误差以95％的显着性水平绘制，可以由用户修改。

.. 注意::

   使用估计的误差协方差矩阵 :math:`\hat \Sigma_u` 的 Cholesky 分解完成正交化，因此解释可能会根据变量顺序而变化

.. ipython:: python

   @savefig var_irf.png
   irf.plot(orth=False)


请注意 `plot` 函数是灵活的，并且可以根据需要来绘制感兴趣的变量：:

.. ipython:: python

   @savefig var_realgdp.png
   irf.plot(impulse='realgdp')

累积效应 :math:`\Psi_n = \sum_{i=0}^n \Phi_i` 可以用长期效应进行绘制，如下所示

.. ipython:: python

   @savefig var_irf_cum.png
   irf.plot_cum_effects(orth=False)


.. currentmodule:: statsmodels.tsa.vector_ar.irf
.. autosummary::
   :toctree: generated/

   IRAnalysis

预测误差方差分解 (FEVD)
--------------------------------------------

可以使用正交化的脉冲响应 :math:`\Theta_i` 来分解提前 i-step 预测中的分量j在k上的预测误差:

.. math::

    \omega_{jk, i} = \sum_{i=0}^{h-1} (e_j^\prime \Theta_i e_k)^2 / \mathrm{MSE}_j(h)

    \mathrm{MSE}_j(h) = \sum_{i=0}^{h-1} e_j^\prime \Phi_i \Sigma_u \Phi_i^\prime e_j

通过 `fevd` 函数计算的，总共需要经过以下步骤:

.. ipython:: python

   fevd = results.fevd(5)
   fevd.summary()

它们也可以通过返回的 :class:`FEVD` 对象可视化:

.. ipython:: python

   @savefig var_fevd.png
   results.fevd(20).plot()


.. currentmodule:: statsmodels.tsa.vector_ar.var_model
.. autosummary::
   :toctree: generated/

   FEVD

统计检验
-----------------

提供了许多不同的方法来执行关于模型结果以及模型假设的有效性的假设检验 (正态化、
whiteness / "iid-ness"  误差, 等等.).

Granger 因果关系
~~~~~~~~~~~~~~~~~

对于 "causal"的某种定义，人们通常会对一个变量或一组变量是否对另一个变量 "causal" 感兴趣。在 VAR
模型的上下文中，可以说一组变量是其中一个 VAR 方程中的 Granger 因果关系. 我们不会详细介绍 Granger
因果关系的数学或定义, 而是将其留给读者。 :class:`VARResults` 对象具有用于执行任一 Wald (:math:`\chi^2`) 测试或F-检验.

.. ipython:: python

   results.test_causality('realgdp', ['realinv', 'realcons'], kind='f')

正态化
~~~~~~~~~

如本文档开头所指出的，假定白噪声分量 :math:`u_t` 是正态分布的。尽管参数估计值一致或渐近正常不需要此假设，
但当残差为高斯白噪声时，在有限样本中，结果通常更可靠。为了测试此假设是否与数据集一致 :class:`VARResults` 提供了 `test_normality` 方法.

.. ipython:: python

    results.test_normality()

残差白度
~~~~~~~~~~~~~~~~~~~~~~

为了测试估计残差的白度 (这意味着不存在显著残差自相关的) 可以使用 :class:`VARResults` 类的方法`test_whiteness` 。


.. currentmodule:: statsmodels.tsa.vector_ar.hypothesis_test_results
.. autosummary::
   :toctree: generated/

   HypothesisTestResults
   CausalityTestResults
   NormalityTestResults
   WhitenessTestResults

.. _svar:

结构矢量自回归
---------------------------------

有一组匹配的类可以处理某些类型的结构化 VAR 模型.

.. module:: statsmodels.tsa.vector_ar.svar_model
   :synopsis: Structural vector autoregressions and related tools

.. currentmodule:: statsmodels.tsa.vector_ar.svar_model

.. autosummary::
   :toctree: generated/

   SVAR
   SVARProcess
   SVARResults

.. _vecm:

矢量误差校正模型 (VECM)
-------------------------------------

矢量误差校正模型用于研究与一个或多个永久随机趋势 (单位根)的短期偏差。VECM 通过施加
假设的随机趋势数量所隐含的结构来建模时间序列矢量的差异。 class:`VECM` 用于指定和估计这些模型。

 VECM(:math:`k_{ar}-1`) 具有以下形式

.. math::

    \Delta y_t = \Pi y_{t-1} + \Gamma_1 \Delta y_{t-1} + \ldots
                   + \Gamma_{k_{ar}-1} \Delta y_{t-k_{ar}+1} + u_t

其中

.. math::

    \Pi = \alpha \beta'

 如第7章中 [1]_ 的所述。

具有确定性项的 VECM(:math:`k_{ar} - 1`) 具有以下形式

.. math::

   \Delta y_t = \alpha \begin{pmatrix}\beta' & \eta'\end{pmatrix} \begin{pmatrix}y_{t-1} \\
                D^{co}_{t-1}\end{pmatrix} + \Gamma_1 \Delta y_{t-1} + \dots + \Gamma_{k_{ar}-1} \Delta y_{t-k_{ar}+1} + C D_t + u_t.

在 :math:`D^{co}_{t-1}` 中，我们具有位于协整关系内（或仅限于协整关系）的确定项。
:math:`\eta` 是对应的估计量。要在协整关系中传递确定项, 我们可以使用 `exog_coint` 参数。
对于截距和线性趋势这两种特殊情况，存在一种更简单的方式来表明这些内容: 我们可以将 ``"ci"`` 和 ``"li"``
分别传递给 `deterministic` 参数. 因此，对于协整关系的内部截距，我们可以传递 ``"ci"`` 作为 `deterministic` 参数
或者传递 `np.ones(len(data))` 作为 `exog_coint` 参数，如果 `data` 以
`endog` 参数来传递. 对于所有的 :math:`t` 可以确定 :math:`D_{t-1}^{co} = 1` 。


我们还可以在协整关系之外使用确定项。这些定义的 :math:`D_t` ，在上面的公式中， :math:`C` 具有相应的估计量。
我们通过传递他们给 `exog` 参数来指定确定项。对于截距 and/or 线性趋势，我们可以再次使用确定性的可能性。
对于截距我们可以传递 ``"co"`` ，且对于线性趋势我们可以传递 ``"lo"`` ，其中 `o` 表示 `outside`。

下表表示了在 [2]_ 中考虑的五种情况， 最后一列表示在每种情况下要传递给 `deterministic` 参数的字符串。

====  ===============================  ===================================  =============
Case         截距                                线性趋势的斜率              `deterministic`
====  ===============================  ===================================  =============
I     0                                0                                    ``"nc"``
II    :math:`- \alpha \beta^T \mu`     0                                    ``"ci"``
III   :math:`\neq 0`                   0                                    ``"co"``
IV    :math:`\neq 0`                   :math:`- \alpha \beta^T \gamma`      ``"coli"``
V     :math:`\neq 0`                   :math:`\neq 0`                       ``"colo"``
====  ===============================  ===================================  =============

.. currentmodule:: statsmodels.tsa.vector_ar.vecm
.. autosummary::
   :toctree: generated/

   VECM
   coint_johansen
   JohansenTestResult
   select_order
   select_coint_rank
   VECMResults
   CointRankResults


参考文献
----------
.. [1] Lütkepohl, H. 2005. *多元时间序列分析的新概论*. Springer.

.. [2] Johansen, S. 1995. *Likelihood-Based Inference in Cointegrated *
       *Vector Autoregressive Models*. Oxford University Press.
