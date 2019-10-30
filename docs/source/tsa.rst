.. module:: statsmodels.tsa
   :synopsis: Time-series analysis

.. currentmodule:: statsmodels.tsa


.. _tsa:


时间序列分析 :mod:`tsa`
===============================

:mod:`statsmodels.tsa` 包含对时间序列分析有用的模型类和函数。基本模型包括单变量自回归模型（AR），
矢量自回归模型（VAR）和单变量自回归移动平均模型（ARMA）。非线性模型包括马尔可夫切换动态回归和自回归。
它还包括时间序列的描述性统计数据，例如自相关，部分自相关函数和周期图，以及ARMA或相关过程的相应理论属性。
它还包括使用自回归和移动平均滞后多项式的方法。此外，还提供相关的统计测试和一些有用的辅助功能。

使用卡尔曼滤波或直接滤波，通过精确的或有条件的最大似然或有条件的最小二乘来进行估计。

当前，必须从相应的模块导入函数和类，但是主要类将在statsmodels.tsa命名空间中提供。
模块结构在statsmodels.tsa中

- stattools : 经验特性和检验, acf, pacf, granger-causality,
  adf 单位根检验, kpss 检验, bds 检验, ljung-box 检验等.
- ar_model : 单变量自回归过程，具有条件和精确的最大似然以及条件最小二乘的估计。
- arima_model : 单变量ARMA过程，具有条件和精确最大似然以及条件最小二乘的估计。
- statespace : 全面的状态空间模型规范和估计。请参阅状态 :ref:`statespace documentation <statespace>`.
- vector_ar, var : 矢量自回归过程（VAR）和矢量误差校正模型，估计，脉冲响应分析，预测误差方差分解和
数据可视化工具。请参阅 :ref:`vector_ar documentation <var>`.
- kalmanf : 使用卡尔曼滤波的ARMA和其他具有精确MLE的模型的估计类
- arma_process : 具有给定参数的arma进程的属性，包括用于在ARMA，MA和AR表示以及acf，pacf，频谱密度，
脉冲响应函数等之间进行转换的工具
- sandbox.tsa.fftarma : 与arma_process类似，但在频域中工作
- tsatools : 附加的辅助函数，用于创建滞后变量的数组，构造趋势，下降趋势等的回归变量。
- filters : 用于过滤时间序列的辅助功能
- regime_switching : 马尔可夫转换的动态回归和自回归模型

statsmodel的其他部分还包含一些对时间序列分析有用的其他功能，例如其他统计测试。

matplotlib，nitime和scikits.talkbox中也提供了一些相关功能。这些功能更多地设计用于信号处理，
在这些信号处理中可获得更长的时间序列，并且在频域中工作更多。


.. currentmodule:: statsmodels.tsa


描述性统计和检验
""""""""""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   stattools.acovf
   stattools.acf
   stattools.pacf
   stattools.pacf_yw
   stattools.pacf_ols
   stattools.pacf_burg
   stattools.ccovf
   stattools.ccf
   stattools.periodogram
   stattools.adfuller
   stattools.kpss
   stattools.zivot_andrews
   stattools.coint
   stattools.bds
   stattools.q_stat
   stattools.grangercausalitytests
   stattools.levinson_durbin
   stattools.innovations_algo
   stattools.innovations_filter
   stattools.levinson_durbin_pacf
   stattools.arma_order_select_ic
   x13.x13_arima_select_order
   x13.x13_arima_analysis

估计
""""""""""

以下是主要的估算类，可以通过statsmodels.tsa.api及其结果类进行访问

单变量自回归过程 (AR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   ar_model.AR
   ar_model.ARResults


自回归移动平均过程（ARMA）和卡尔曼滤波
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.tsa

对于大多数用户而言，应作为起点的基本ARIMA模型和结果类是:

.. autosummary::
   :toctree: generated/

   arima_model.ARMA
   arima_model.ARMAResults
   arima_model.ARIMA
   arima_model.ARIMAResults

可用于计算ARMA类型模型的对数似然函数的一些高级基础底层类和函数包括
（请注意，最终用户很少需要这些）:

.. autosummary::
   :toctree: generated/

   kalmanf.kalmanfilter.KalmanFilter
   innovations.arma_innovations.arma_innovations
   innovations.arma_innovations.arma_loglike
   innovations.arma_innovations.arma_loglikeobs
   innovations.arma_innovations.arma_score
   innovations.arma_innovations.arma_scoreobs


指数平滑
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   holtwinters.ExponentialSmoothing
   holtwinters.SimpleExpSmoothing
   holtwinters.Holt
   holtwinters.HoltWintersResults


ARMA 程序
""""""""""""

以下是用于给定滞后多项式的ARMA程序的理论属性的工具.

.. autosummary::
   :toctree: generated/

   arima_process.ArmaProcess
   arima_process.ar2arma
   arima_process.arma2ar
   arima_process.arma2ma
   arima_process.arma_acf
   arima_process.arma_acovf
   arima_process.arma_generate_sample
   arima_process.arma_impulse_response
   arima_process.arma_pacf
   arima_process.arma_periodogram
   arima_process.deconvolve
   arima_process.index2lpol
   arima_process.lpol2index
   arima_process.lpol_fiar
   arima_process.lpol_fima
   arima_process.lpol_sdiff

.. currentmodule:: statsmodels.sandbox.tsa.fftarma

.. autosummary::
   :toctree: generated/

   ArmaFft

.. currentmodule:: statsmodels.tsa

Statespace 模型
"""""""""""""""""
See the :ref:`statespace documentation. <statespace>`


Vector ARs and Vector Error Correction Models
"""""""""""""""""""""""""""""""""""""""""""""
See the :ref:`vector_ar documentation. <var>`

Regime switching 模型
"""""""""""""""""""""""

.. currentmodule:: statsmodels.tsa.regime_switching.markov_regression
.. autosummary::
   :toctree: generated/

   马尔可夫回归

.. currentmodule:: statsmodels.tsa.regime_switching.markov_autoregression
.. autosummary::
   :toctree: generated/

   马尔可夫自回归


时间序列过滤器
"""""""""""""""""""

.. currentmodule:: statsmodels.tsa.filters.bk_filter
.. autosummary::
   :toctree: generated/

   bkfilter

.. currentmodule:: statsmodels.tsa.filters.hp_filter
.. autosummary::
   :toctree: generated/

   hpfilter

.. currentmodule:: statsmodels.tsa.filters.cf_filter
.. autosummary::
   :toctree: generated/

   cffilter

.. currentmodule:: statsmodels.tsa.filters.filtertools
.. autosummary::
   :toctree: generated/

   convolution_filter
   recursive_filter
   miso_lfilter
   fftconvolve3
   fftconvolveinv


.. currentmodule:: statsmodels.tsa.seasonal
.. autosummary::
   :toctree: generated/

   季节性分解
   STL
   DecomposeResult

TSA 工具
"""""""""

.. currentmodule:: statsmodels.tsa.tsatools

.. autosummary::
   :toctree: generated/

   add_lag
   add_trend
   detrend
   lagmat
   lagmat2ds

VARMA 流程
"""""""""""""

.. currentmodule:: statsmodels.tsa.varma_process
.. autosummary::
   :toctree: generated/

   VarmaPoly

插补
"""""""""""""

.. currentmodule:: statsmodels.tsa.interp.denton
.. autosummary::
   :toctree: generated/

   dentonm
