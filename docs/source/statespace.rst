.. module:: statsmodels.tsa.statespace
   :synopsis: Statespace models for time-series analysis

.. currentmodule:: statsmodels.tsa.statespace


.. _statespace:


基于状态空间方法的时间序列分析 :mod:`statespace`
=============================================================

:mod:`statsmodels.tsa.statespace` 包含类和函数，这些类和函数对于使用状态空间方法进行时间序列分析很有用。

一般状态空间模型的形式为

.. math::

  y_t & = Z_t \alpha_t + d_t + \varepsilon_t \\
  \alpha_t & = T_t \alpha_{t-1} + c_t + R_t \eta_t \\

其中 :math:`y_t` 指的是观测矢量在时间 :math:`t`,
:math:`\alpha_t` 指（未观察到）状态向量在时间 
:math:`t`, 并且其中所述不规则要素被定义为

.. math::

  \varepsilon_t \sim N(0, H_t) \\
  \eta_t \sim N(0, Q_t) \\

方程中的其余变量 (:math:`Z_t, d_t, H_t, T_t, c_t, R_t, Q_t`) 是描述过程的矩阵。
它们的变量名称和尺寸如下

Z : `design`          :math:`(k\_endog \times k\_states \times nobs)`

d : `obs_intercept`   :math:`(k\_endog \times nobs)`

H : `obs_cov`         :math:`(k\_endog \times k\_endog \times nobs)`

T : `transition`      :math:`(k\_states \times k\_states \times nobs)`

c : `state_intercept` :math:`(k\_states \times nobs)`

R : `selection`       :math:`(k\_states \times k\_posdef \times nobs)`

Q : `state_cov`       :math:`(k\_posdef \times k\_posdef \times nobs)`

在其中一个矩阵是时间不变的情况下 (例如, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), 
其最后一个维度可能是尺寸为 :math:`1` 而不是尺寸为 `nobs`.

这种通用方式封装了许多最常用的线性时间序列模型 (请参见下文) 并且非常灵活，
允许在缺少观测值，预测，脉冲响应函数等情况下进行估算。

**示例: AR(2) 模型**

自回归模型是将模型置于状态空间形式的一个很好的入门示例。回想一下AR（2）模型通常写为:

.. math::

   y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \epsilon_t

可以通过以下方式将其放入状态空间形式:

.. math::

   y_t & = \begin{bmatrix} 1 & 0 \end{bmatrix} \alpha_t \\
   \alpha_t & = \begin{bmatrix}
      \phi_1 & \phi_2 \\
           1 &      0
   \end{bmatrix} \alpha_{t-1} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t

其中

.. math::

   Z_t \equiv Z = \begin{bmatrix} 1 & 0 \end{bmatrix}

和

.. math::

   T_t \equiv T & = \begin{bmatrix}
      \phi_1 & \phi_2 \\
           1 &      0
   \end{bmatrix} \\
   R_t \equiv R & = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\
   \eta_t & \sim N(0, \sigma^2)

此模型中有三个未知参数:
:math:`\phi_1, \phi_2, \sigma^2`.

模型和估算
---------------------

以下是主要的估算类，可以通过 `statsmodels.tsa.statespace.api` 及其结果类进行访问 。

具有外生回归变量的季节性自回归综合移动平均数 (SARIMAX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 `SARIMAX` 类 是使用用于估计状态空间后端创建的完全成熟的模型的一个例子. `SARIMAX` 使用起来与
:ref:`tsa <tsa>` 模型非常的相似, 但是可以通过添加相加和相乘季节效应的估计以及
任意趋势多项式来在更广泛的模型上使用。

.. autosummary::
   :toctree: generated/

   sarimax.SARIMAX
   sarimax.SARIMAXResults

有关使用此模型的示例，请参见
`SARIMAX example notebook <examples/notebooks/generated/statespace_sarimax_stata.html>`__
或以下非常简短的代码段:


.. code-block:: python

   # 加载 statsmodels api
   import statsmodels.api as sm

   # 加载数据集 
   endog = pd.read_csv('your/dataset/here.csv')

   # 我们可以适应的AR（2）模型，表述为 
   mod_ar2 = sm.tsa.SARIMAX(endog, order=(2,0,0))
   # 请注意，mod_ar2是SARIMAX类的一个实例

   # 通过最大似然拟合模型 
   res_ar2 = mod_ar2.fit()
   # 注意res_ar2是SARIMAXResults类的实例

   # 显示结果摘要
   print(res_ar2.summary())

   # 我们还可以使用季节因素来拟合更复杂的模型。
   # 作为例子， SARIMA(1,1,1) x (0,1,1,4):
   mod_sarimax = sm.tsa.SARIMAX(endog, order=(1,1,1),
                                seasonal_order=(0,1,1,4))
   res_sarimax = mod_sarimax.fit()

   # 显示 summary 的结果
   print(res_sarimax.summary())

结果对象具有其他statsmodels结果对象期望的许多属性和方法，包括 standard errors（标准差）, z-statistics（z 统计量）,
和 prediction / forecasting.

在后台,  `SARIMAX` 模型根据模型规范创建设计和过渡矩阵（有时还包括其他一些矩阵）。

未观察到的组件
^^^^^^^^^^^^^^^^^^^^^

 `UnobservedComponents` 类是状态空间模型的另一个示例。

.. autosummary::
   :toctree: generated/

   structural.UnobservedComponents
   structural.UnobservedComponentsResults

有关使用此模型的示例，请参见 `example notebook <examples/notebooks/generated/statespace_structural_harvey_jaeger.html>`__ 
使用未观察到的组件模型将 `decompose a time series into a trend and cycle <examples/notebooks/generated/statespace_cycles.html>`__ 
或下面的非常简短的代码片段：

.. code-block:: python

   # 加载 statsmodels api
   import statsmodels.api as sm

   # 加载数据集 
   endog = pd.read_csv('your/dataset/here.csv')

   # 拟合本地模型 
   mod_ll = sm.tsa.UnobservedComponents(endog, 'local level')
   # 注意mod_ll是UnobservedComponents类的实例

   # 通过最大似然拟合模型 
   res_ll = mod_ll.fit()
   # 注意res_ll是UnobservedComponentsResults类的实例

   # 显示 summary 的结果
   print(res_ll.summary())

   # 显示估计水平和趋势成分系列的图
   fig_ll = res_ll.plot_components()

   # 我们可以进一步添加一个阻尼随机周期，如下所示 
   mod_cycle = sm.tsa.UnobservedComponents(endog, 'local level', cycle=True,
                                           damped_cycle=true,
                                           stochastic_cycle=True)
   res_cycle = mod_cycle.fit()

   # 显示 summary 的结果
   print(res_cycle.summary())

   # 显示估计水平，趋势和周期成分系列的图 
   fig_cycle = res_cycle.plot_components()

矢量自回归移动平均与异源回归变量 (VARMAX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 `VARMAX` 类是多元状态空间模型的一个示例。

.. autosummary::
   :toctree: generated/

   varmax.VARMAX
   varmax.VARMAXResults

有关使用此模型的示例，请参见 `VARMAX example notebook <examples/notebooks/generated/statespace_varmax.html>`__ 或非常简短的代码段：

.. code-block:: python

   # 加载 statsmodels api
   import statsmodels.api as sm

   # 加载（多变量）数据集
   endog = pd.read_csv('your/dataset/here.csv')

   # 拟合模型 
   mod_var1 = sm.tsa.VARMAX(endog, order=(1,0))
   # Note that mod_var1 is an instance of the VARMAX class

   # 通过最大似然拟合模型 
   res_var1 = mod_var1.fit()
   # 注意res_var1是VARMAXResults类的实例

   # 显示 summary 的结果
   print(res_var1.summary())

   # 构造脉冲响应 
   irfs = res_ll.impulse_responses(steps=10)

动态因素模型
^^^^^^^^^^^^^^^^^^^^^

 `DynamicFactor` 类是多元状态空间模型的另一个示例。

.. autosummary::
   :toctree: generated/

   dynamic_factor.DynamicFactor
   dynamic_factor.DynamicFactorResults

有关使用此模型的示例，请参见 `Dynamic Factor example notebook <examples/notebooks/generated/statespace_dfm_coincident.html>`__ 或以下非常简短的代码段：
.. code-block:: python

   # 加载 statsmodels api
   import statsmodels.api as sm

   # 加载数据集
   endog = pd.read_csv('your/dataset/here.csv')

   # 拟合模型
   mod_dfm = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=2)
   # 注意mod_dfm是DynamicFactor类的实例

   # 通过最大似然拟合模型 
   res_dfm = mod_dfm.fit()
   # 注意res_dfm是DynamicFactorResults类的实例


   # 显示 summary 的结果
   print(res_ll.summary())

   # 显示各个估计因子对 endog 的回归预测的 r 方值图
   # individual estimated factors on endogenous variables.
   fig_dfm = res_ll.plot_coefficients_of_determination()

自定义状态空间模型
^^^^^^^^^^^^^^^^^^^^^^^^^

状态空间模型的真正功能是允许创建和估计自定义模型。通常，这是通过扩展以下两个类来完成的，
这些类捆绑了所有状态空间表示，卡尔曼滤波以及用于估计和结果输出的最大似然拟合功能。

.. autosummary::
   :toctree: generated/

   mlemodel.MLEModel
   mlemodel.MLEResults

有关演示如何创建和估计自定义状态空间模型的基本示例，请参见 `Local Linear Trend example notebook <examples/notebooks/generated/statespace_local_linear_trend.html>`__.
有关更复杂的示例，请参见 `SARIMAX` 和
`SARIMAXResults` 类的源代码，它们是通过扩展 `MLEModel` 和
`MLEResults` 构建的

在简单的情况下，完全可以使用MLEModel类来构造模型。例如，可以使用以下代码来构造和估算上面的AR（2）模型:

.. code-block:: python

   import numpy as np
   from scipy.signal import lfilter
   import statsmodels.api as sm

   # 真实的模型参数
   nobs = int(1e3)
   true_phi = np.r_[0.5, -0.2]
   true_sigma = 1**0.5

   # 模拟的时间序列
   np.random.seed(1234)
   disturbances = np.random.normal(0, true_sigma, size=(nobs,))
   endog = lfilter([1], np.r_[1, -true_phi], disturbances)

   # 构建模型
   class AR2(sm.tsa.statespace.MLEModel):
       def __init__(self, endog):
           # 初始化状态空间模型
           super(AR2, self).__init__(endog, k_states=2, k_posdef=1,
                                     initialization='stationary')

           # 设定状态空间的固定部件
           self['design'] = [1, 0]
           self['transition'] = [[0, 0],
                                     [1, 0]]
           self['selection', 0, 0] = 1

       # 描述参数如何输入模型
       def update(self, params, transformed=True, **kwargs):
           params = super(AR2, self).update(params, transformed, **kwargs)

           self['transition', 0, :] = params[:2]
           self['state_cov', 0, 0] = params[2]

       # 指定启动参数和参数名
       @property
       def start_params(self):
           return [0,0,1]  # these are very simple

   # 创建并拟合模型
   mod = AR2(endog)
   res = mod.fit()
   print(res.summary())

输出 summary 的结果表::

                              Statespace Model Results                           
   ==============================================================================
   Dep. Variable:                      y   No. Observations:                 1000
   Model:                            AR2   Log Likelihood               -1389.437
   Date:                Wed, 26 Oct 2016   AIC                           2784.874
   Time:                        00:42:03   BIC                           2799.598
   Sample:                             0   HQIC                          2790.470
                                  - 1000                                         
   Covariance Type:                  opg                                         
   ==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
   ------------------------------------------------------------------------------
   param.0        0.4395      0.030     14.730      0.000       0.381       0.498
   param.1       -0.2055      0.032     -6.523      0.000      -0.267      -0.144
   param.2        0.9425      0.042     22.413      0.000       0.860       1.025
   ===================================================================================
   Ljung-Box (Q):                       24.25   Jarque-Bera (JB):                 0.22
   Prob(Q):                              0.98   Prob(JB):                         0.90
   Heteroskedasticity (H):               1.05   Skew:                            -0.04
   Prob(H) (two-sided):                  0.66   Kurtosis:                         3.02
   ===================================================================================
   
   Warnings:
   [1] Covariance matrix calculated using the outer product of gradients (complex-step).

结果对象具有其他statsmodels结果对象期望的许多属性和方法，包括 standard errors（标准差）, z-statistics（z 统计量）,
and prediction / forecasting.

可以使用更高级的用法，包括指定参数转换和为更详尽的输出摘要指定参数名称。

状态空间表示和卡尔曼滤波
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

尽管自定义模型的创建几乎总是通过扩展`MLEModel` 和 `MLEResults` 来完成的, 
但了解这些类背后的上层结构可能很有用。.

最大似然估计需要评估模型的似然函数，对于状态空间形式的模型，似然函数作为运行卡尔曼滤波的副产品进行评估

 `MLEModel` 使用两个类来简化状态空间模型 和 Kalman 过滤的规范: `Representation` 和 `KalmanFilter`.

 `Representation` 类是定义状态空间模型表示的部分。简单来说，它保存状态空间矩阵
(`design`, `obs_intercept`, etc.; s请参见上面的状态空间模型简介) 并允许对其进行操作。

`FrozenRepresentation` 是最基本的结果类型类，因为它在任何给定时间获取状态空间表示的
"snapshot" 。有关可用属性的完整列表，请参见类文档。

.. autosummary::
   :toctree: generated/

   representation.Representation
   representation.FrozenRepresentation

 `KalmanFilter` 类是 Representation 类的子类，提供过滤功能. 一旦构造了状态空间表示矩阵， 就可以调用
 :py:meth:`filter <kalman_filter.KalmanFilter.filter>`
方法，从而生成一个 `FilterResults` 实例; `FilterResults` 类是 `FrozenRepresentation` 类的子类.

 `FilterResults` c类不仅保存状态空间模型的冻结表示 ( design 设计、 transition 转换 等矩阵以及模型尺寸等)
  而且还保存过滤输出，包括
:py:attr:`filtered state <kalman_filter.FilterResults.filtered_state>` 和
loglikelihood (有关可用结果的完整列表，请参见类文档). 它还提供了一种
:py:meth:`predict <kalman_filter.FilterResults.predict>` 方法，该方法可以进行样本内预测或样本外预测.
 A similar method,
:py:meth:`predict <kalman_filter.FilterResults.get_prediction>`, 可提供其他预测或预测结果，包括置信区间。

.. autosummary::
   :toctree: generated/

   kalman_filter.KalmanFilter
   kalman_filter.FilterResults
   kalman_filter.PredictionResults

 `KalmanSmoother` 类是提供平滑性能的 `KalmanFilter` 类的子类，一旦构造了状态空间表示矩阵， 
 就可以调用 :py:meth:`filter <kalman_filter.KalmanSmoother.smooth>`
方法，从而生成一个 `SmootherResults` 实例; `SmootherResults` 类是 `FilterResults` 类的子类.

 `SmootherResults` 类拥有 `FilterResults` 类的所有输出, 还包括平滑输出，包括
:py:attr:`smoothed state <kalman_filter.SmootherResults.smoothed_state>` 和
loglikelihood (有关可用结果的完整列表，请参见类文档).在时间 `t` 处， "filtered" 输出
是指以直到时间t的观察为条件的估计 `t`,而 "smoothed" 输出是指以数据集中的整个观察集为条件的估计.

.. autosummary::
   :toctree: generated/

   kalman_smoother.KalmanSmoother
   kalman_smoother.SmootherResults

状态空间诊断
----------------------

在估计任何状态空间模型（无论是内置的还是自定义的）之后，可以使用三种诊断测试来帮助评估模型是否符合基础的统计假设。这些测试是：

- :py:meth:`test_normality <mlemodel.MLEResults.test_normality>`
- :py:meth:`test_heteroskedasticity <mlemodel.MLEResults.test_heteroskedasticity>`
- :py:meth:`test_serial_correlation <mlemodel.MLEResults.test_serial_correlation>`

出于相同目的，可以使用许多回归残差的标准图。这些可以使用命令
:py:meth:`plot_diagnostics <mlemodel.MLEResults.plot_diagnostics>` 产生。

状态空间工具
----------------

状态空间建模或SARIMAX类使用多种工具:

.. autosummary::
   :toctree: generated/

   tools.companion_matrix
   tools.diff
   tools.is_invertible
   tools.constrain_stationary_univariate
   tools.unconstrain_stationary_univariate
   tools.constrain_stationary_multivariate
   tools.unconstrain_stationary_multivariate
   tools.validate_matrix_shape
   tools.validate_vector_shape
