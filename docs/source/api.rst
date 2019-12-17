API 参考
=============
主要的 statsmodels API 分为以下模型:

* ``statsmodels.api``: 横截面模型和方法。使用规范地导入 ``import statsmodels.api as sm``.
* ``statsmodels.tsa.api``: 时间序列模型和方法。使用规范地导入 ``import statsmodels.tsa.api as tsa``.
* ``statsmodels.formula.api``: 一个方便的界面用于使用公式字符串和 DataFrames 来指定模型。
该 API 直接公开 ``from_formula`` 支持公式 API 模型的类方法。规范地使用  ``import statsmodels.formula.api as smf``

.. autosummary::
API 专注于模型和最常用的统计测试以及工具。 `Import Paths and Structure`_ 介绍了两个 API 模块的设计，
以及从 API 导入与直接从定义模型的模块直接导入有何不同。有关模型的使用，统计信息和工具的完整列表，请参见 :ref:`user-guide:User Guide` 中的详细主题页面。

``statsmodels.api``
-------------------

回归
~~~~~~~~~~
.. autosummary::

   ~statsmodels.regression.linear_model.OLS
   ~statsmodels.regression.linear_model.GLS
   ~statsmodels.regression.linear_model.GLSAR
   ~statsmodels.regression.linear_model.WLS
   ~statsmodels.regression.recursive_ls.RecursiveLS
   ~statsmodels.regression.rolling.RollingOLS
   ~statsmodels.regression.rolling.RollingWLS

插补
~~~~~~~~~~
.. autosummary::

   ~statsmodels.imputation.bayes_mi.BayesGaussMI
   ~statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM
   ~statsmodels.multivariate.factor.Factor
   ~statsmodels.imputation.bayes_mi.MI
   ~statsmodels.imputation.mice.MICE
   ~statsmodels.imputation.mice.MICEData

广义估计方程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.genmod.generalized_estimating_equations.GEE
   ~statsmodels.genmod.generalized_estimating_equations.NominalGEE
   ~statsmodels.genmod.generalized_estimating_equations.OrdinalGEE

广义线性模型
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.genmod.generalized_linear_model.GLM
   ~statsmodels.gam.generalized_additive_model.GLMGam
   ~statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM

离散和计数模型
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.discrete.discrete_model.GeneralizedPoisson
   ~statsmodels.discrete.discrete_model.Logit
   ~statsmodels.discrete.discrete_model.MNLogit
   ~statsmodels.discrete.discrete_model.Poisson
   ~statsmodels.discrete.discrete_model.Probit
   ~statsmodels.discrete.discrete_model.NegativeBinomial
   ~statsmodels.discrete.discrete_model.NegativeBinomialP
   ~statsmodels.discrete.count_model.ZeroInflatedGeneralizedPoisson
   ~statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP
   ~statsmodels.discrete.count_model.ZeroInflatedPoisson

多元模型
~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.multivariate.manova.MANOVA
   ~statsmodels.multivariate.pca.PCA

混合模型
~~~~~~~~~~~
.. autosummary::

   ~statsmodels.regression.mixed_linear_model.MixedLM
   ~statsmodels.duration.hazard_regression.PHReg
   ~statsmodels.regression.quantile_regression.QuantReg
   ~statsmodels.robust.robust_linear_model.RLM
   ~statsmodels.duration.survfunc.SurvfuncRight


绘图
~~~~~~~~
.. autosummary::

   ~statsmodels.graphics.gofplots.ProbPlot
   ~statsmodels.graphics.gofplots.qqline
   ~statsmodels.graphics.gofplots.qqplot
   ~statsmodels.graphics.gofplots.qqplot_2samples

工具
~~~~~
.. autosummary::

   ~statsmodels.__init__.test
   ~statsmodels.tools.tools.add_constant
   ~statsmodels.tools.tools.categorical
   ~statsmodels.iolib.smpickle.load_pickle
   ~statsmodels.tools.print_version.show_versions
   ~statsmodels.tools.web.webdoc


``statsmodels.tsa.api``
-----------------------

统计和检验
~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.stattools.acf
   ~statsmodels.tsa.stattools.acovf
   ~statsmodels.tsa.stattools.adfuller
   ~statsmodels.tsa.stattools.bds
   ~statsmodels.tsa.stattools.ccf
   ~statsmodels.tsa.stattools.ccovf
   ~statsmodels.tsa.stattools.coint
   ~statsmodels.tsa.stattools.kpss
   ~statsmodels.tsa.stattools.pacf
   ~statsmodels.tsa.stattools.pacf_ols
   ~statsmodels.tsa.stattools.pacf_yw
   ~statsmodels.tsa.stattools.periodogram
   ~statsmodels.tsa.stattools.q_stat

单变量时间序列分析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.ar_model.AR
   ~statsmodels.tsa.arima_model.ARIMA
   ~statsmodels.tsa.arima_model.ARMA
   ~statsmodels.tsa.statespace.sarimax.SARIMAX
   ~statsmodels.tsa.stattools.arma_order_select_ic
   ~statsmodels.tsa.arima_process.arma_generate_sample
   ~statsmodels.tsa.arima_process.ArmaProcess

指数平滑
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.holtwinters.ExponentialSmoothing
   ~statsmodels.tsa.holtwinters.Holt
   ~statsmodels.tsa.holtwinters.SimpleExpSmoothing


多元模型
~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.statespace.dynamic_factor.DynamicFactor
   ~statsmodels.tsa.vector_ar.var_model.VAR
   ~statsmodels.tsa.statespace.varmax.VARMAX
   ~statsmodels.tsa.vector_ar.svar_model.SVAR
   ~statsmodels.tsa.vector_ar.vecm.VECM
   ~statsmodels.tsa.statespace.structural.UnobservedComponents

过滤和分解
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.seasonal.seasonal_decompose
   ~statsmodels.tsa.seasonal.STL
   ~statsmodels.tsa.filters.bk_filter.bkfilter
   ~statsmodels.tsa.filters.cf_filter.cffilter
   ~statsmodels.tsa.filters.hp_filter.hpfilter

Markov Regime Switching 模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression
   ~statsmodels.tsa.regime_switching.markov_regression.MarkovRegression

时间序列工具
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.tsatools.add_lag
   ~statsmodels.tsa.tsatools.add_trend
   ~statsmodels.tsa.tsatools.detrend
   ~statsmodels.tsa.tsatools.lagmat
   ~statsmodels.tsa.tsatools.lagmat2ds

X12/X13 接口
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.x13.x13_arima_analysis
   ~statsmodels.tsa.x13.x13_arima_select_order

``statsmodels.formula.api``
---------------------------

模型
~~~~~~

公式API中公开的方法的功能描述是通用的。有关详细信息，请参见父模型的文档。

.. autosummary::
   :toctree: generated/

   ~statsmodels.formula.api.gls
   ~statsmodels.formula.api.wls
   ~statsmodels.formula.api.ols
   ~statsmodels.formula.api.glsar
   ~statsmodels.formula.api.mixedlm
   ~statsmodels.formula.api.glm
   ~statsmodels.formula.api.rlm
   ~statsmodels.formula.api.mnlogit
   ~statsmodels.formula.api.logit
   ~statsmodels.formula.api.probit
   ~statsmodels.formula.api.poisson
   ~statsmodels.formula.api.negativebinomial
   ~statsmodels.formula.api.quantreg
   ~statsmodels.formula.api.phreg
   ~statsmodels.formula.api.ordinal_gee
   ~statsmodels.formula.api.nominal_gee
   ~statsmodels.formula.api.gee
   ~statsmodels.formula.api.glmgam


.. _importpaths:

导入路径和结构
--------------------------

我们提供了两种从statsmodels导入函数和类的方法：

1. `API import for interactive use`_

   + 允许制表符完成

2. `Direct import for programs`_

   + 避免导入不必要的模块和命令

交互使用的API导入
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于交互式使用，建议的导入为:

.. code-block:: python

    import statsmodels.api as sm

导入 `statsmodels.api` 将加载 statsmodels 大部分公共接口.
这使大多数函数和类在一个或两个级别内方便地可用，而不会使 "sm" 名称空间过于拥挤。

要查看可用的函数和类，可以键入以下内容 (或使用 IPython，Spyder，IDLE 等的名称空间探索功能.):

.. code-block:: python

    >>> dir(sm)
    ['GLM', 'GLS', 'GLSAR', 'Logit', 'MNLogit', 'OLS', 'Poisson', 'Probit', 'RLM',
    'WLS', '__builtins__', '__doc__', '__file__', '__name__', '__package__',
    'add_constant', 'categorical', 'datasets', 'distributions', 'families',
    'graphics', 'iolib', 'nonparametric', 'qqplot', 'regression', 'robust',
    'stats', 'test', 'tools', 'tsa', 'version']

    >>> dir(sm.graphics)
    ['__builtins__', '__doc__', '__file__', '__name__', '__package__',
    'abline_plot', 'beanplot', 'fboxplot', 'interaction_plot', 'qqplot',
    'rainbow', 'rainbowplot', 'violinplot']

    >>> dir(sm.tsa)
    ['AR', 'ARMA', 'SVAR', 'VAR', '__builtins__', '__doc__',
    '__file__', '__name__', '__package__', 'acf', 'acovf', 'add_lag',
    'add_trend', 'adfuller', 'ccf', 'ccovf', 'datetools', 'detrend',
    'filters', 'grangercausalitytests', 'interp', 'lagmat', 'lagmat2ds',
    'pacf', 'pacf_ols', 'pacf_yw', 'periodogram', 'q_stat', 'stattools',
    'tsatools', 'var']

注意
^^^^^

The `api` 模块可能不包括 statsmodels 的所有公共功能。如果您发现应该添加到 api 的内容，
请在 github 上提交问题或将其报告给邮件列表。

statsmodels 的子包包括 `api.py` 模块，这些模块主要用于收集这些子包所需的导入。
例如，将 `subpackage/api.py`
文件导入到 statsmodels.api 中 ::

     from .nonparametric import api as nonparametric

用户不需要直接加载 `subpackage/api.py` 模块。

直接导入程序
~~~~~~~~~~~~~~~~~~~~~~~~~~

``statsmodels`` 子模块按主题排列 (例如，离散选择模型为离散，时间序列分析为 `tsa` ). 
我们的目录树（向下剥离）如下所示：：

    statsmodels/
        __init__.py
        api.py
        discrete/
            __init__.py
            discrete_model.py
            tests/
                results/
        tsa/
            __init__.py
            api.py
            tsatools.py
            stattools.py
            arima_model.py
            arima_process.py
            vector_ar/
                __init__.py
                var_model.py
                tests/
                    results/
            tests/
                results/
        stats/
            __init__.py
            api.py
            stattools.py
            tests/
        tools/
            __init__.py
            tools.py
            decorators.py
            tests/

除了一些用于运行子模块测试的测试代码外，可以大量导入的子模块包含空的 `__init__.py`, 
目的是将所有目录更改为在下一发行版中具有 `api.py` 和空的  `__init__.py` 

导入示例
^^^^^^^^^^^^^^^

函数和类::

    from statsmodels.regression.linear_model import OLS, WLS
    from statsmodels.tools.tools import rank, add_constant

模块 ::

    from statsmodels.datasets import macrodata
    import statsmodels.stats import diagnostic

具有别名的模块 ::

    import statsmodels.regression.linear_model as lm
    import statsmodels.stats.diagnostic as smsdia
    import statsmodels.stats.outliers_influence as oi

我们目前尚无子模块别名的约定。

