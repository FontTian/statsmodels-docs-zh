.. currentmodule:: statsmodels.regression.linear_model


.. _regression:

线性回归
=================

线性模型具有独立且均匀分布的误差，以及具有异方差或自相关的误差。该模块允许通过
普通最小二乘（OLS），加权最小二乘（WLS），广义最小二乘（GLS）和具有自相关AR（p）
误差的可行广义最小二乘进行估计。

有关命令和参数，请查看 `Module Reference模块参考`_ 。

例子
--------

.. ipython:: python

    # 加载模块和数据集
    import numpy as np
    import statsmodels.api as sm
    spector_data = sm.datasets.spector.load(as_pandas=False)
    spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

    # 拟合并输出 OLS 模型
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    res = mod.fit()
    print(res.summary())

更多详细的示例:


* `OLS <examples/notebooks/generated/ols.html>`__
* `WLS <examples/notebooks/generated/wls.html>`__
* `GLS <examples/notebooks/generated/gls.html>`__
* `递归 LS <examples/notebooks/generated/recursive_ls.html>`__
* `Rolling LS <examples/notebooks/generated/rolling_ls.html>`__

技术文档
-----------------------

线性统计模型可以表达为

 :math:`Y = X\beta + \mu`,  其中 :math:`\mu\sim N\left(0,\Sigma\right).`

取决于:math:`\Sigma` 的属性, 目前我们有四种类型可用:

* GLS : 任意协方差的广义最小二乘 :math:`\Sigma` 
* OLS : i.i.d. 误差的普通最小二乘 :math:`\Sigma=\textbf{I}` 
* WLS : 异方误差的加权最小二乘 :math:`\text{diag}\left  (\Sigma\right)` 
* GLSAR : 具有自相关AR（p）误差的可行广义最小二乘法
  :math:`\Sigma=\Sigma\left(\rho\right)` 

所有回归模型都定义了相同的方法并遵循相同的结构，并且可以类似的方式使用。其中一些
包含其他特定于模型的方法和属性。

GLS 是除 RecursiveLS、RollingWLS 和 RollingOLS 之外的其他回归类的超类

.. Class hierachy: TODO

.. yule_walker is not a full model class, but a function that estimate the
.. parameters of a univariate autoregressive process, AR(p). It is used in GLSAR,
.. but it can also be used independently of any models. yule_walker only
.. calculates the estimates and the standard deviation of the lag parameters but
.. not the additional regression statistics. We hope to include yule-walker in
.. future in a separate univariate time series class. A similar result can be
.. obtained with GLSAR if only the constant is included as regressors. In this
.. case the parameter estimates of the lag estimates are not reported, however
.. additional statistics, for example aic, become available.

参考文献
^^^^^^^^^^

回归模型参考常见文献:

* D.C. Montgomery and E.A. Peck. "Introduction to Linear Regression Analysis." 2nd. Ed., Wiley, 1992.

回归模型参考的计量统计学文献:

* R.Davidson and J.G. MacKinnon. "Econometric Theory and Methods," Oxford, 2004.
* W.Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

.. toctree::
..   :maxdepth: 1
..
..   regression_techn1

属性
^^^^^^^^^^

这对于所有回归类模型常见的和详细的描述

pinv_wexog : array
    The `p` x `n` Moore-Penrose pseudoinverse of the whitened design matrix.
    It is approximately equal to
    :math:`\left(X^{T}\Sigma^{-1}X\right)^{-1}X^{T}\Psi`, where
    :math:`\Psi` is defined such that :math:`\Psi\Psi^{T}=\Sigma^{-1}`.
cholsimgainv : array
    The `n` x `n` upper triangular matrix :math:`\Psi^{T}` that satisfies
    :math:`\Psi\Psi^{T}=\Sigma^{-1}`.
df_model : float
    The model degrees of freedom. This is equal to `p` - 1, where `p` is the
    number of regressors. Note that the intercept is not counted as using a
    degree of freedom here.
df_resid : float
    The residual degrees of freedom. This is equal `n - p` where `n` is the
    number of observations and `p` is the number of parameters. Note that the
    intercept is counted as using a degree of freedom here.
llf : float
    The value of the likelihood function of the fitted model.
nobs : float
    The number of observations `n`
normalized_cov_params : array
    A `p` x `p` array equal to :math:`(X^{T}\Sigma^{-1}X)^{-1}`.
sigma : array
    The `n` x `n` covariance matrix of the error terms:
    :math:`\mu\sim N\left(0,\Sigma\right)`.
wexog : array
    The whitened design matrix :math:`\Psi^{T}X`.
wendog : array
    The whitened response variable :math:`\Psi^{T}Y`.

模块参考
----------------

.. module:: statsmodels.regression.linear_model
   :synopsis: Least squares linear models

模型类
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   OLS
   GLS
   WLS
   GLSAR
   yule_walker
   burg

.. module:: statsmodels.regression.quantile_regression
   :synopsis: Quantile regression

.. currentmodule:: statsmodels.regression.quantile_regression

.. autosummary::
   :toctree: generated/

   QuantReg

.. module:: statsmodels.regression.recursive_ls
   :synopsis: Recursive least squares using the Kalman Filter

.. currentmodule:: statsmodels.regression.recursive_ls

.. autosummary::
   :toctree: generated/

   RecursiveLS

.. module:: statsmodels.regression.rolling
   :synopsis: Rolling (moving) least squares

.. currentmodule:: statsmodels.regression.rolling

.. autosummary::
   :toctree: generated/

   RollingWLS
   RollingOLS

.. module:: statsmodels.regression.process_regression
   :synopsis: Process regression

.. currentmodule:: statsmodels.regression.process_regression

.. autosummary::
   :toctree: generated/

   GaussianCovariance
   ProcessMLE

.. module:: statsmodels.regression.dimred
   :synopsis: Dimension reduction methods

.. currentmodule:: statsmodels.regression.dimred

.. autosummary::
   :toctree: generated/

    SlicedInverseReg
    PrincipalHessianDirections
    SlicedAverageVarianceEstimation


结果类
^^^^^^^^^^^^^^^

拟合线性回归模型将返回结果类。与其他线性模型的结果类相比，OLS具有特定的结果类
和一些其他方法。

.. currentmodule:: statsmodels.regression.linear_model

.. autosummary::
   :toctree: generated/

   RegressionResults
   OLSResults
   PredictionResults

.. currentmodule:: statsmodels.base.elastic_net

.. autosummary::
   :toctree: generated/

    RegularizedResults

.. currentmodule:: statsmodels.regression.quantile_regression

.. autosummary::
   :toctree: generated/

   QuantRegResults

.. currentmodule:: statsmodels.regression.recursive_ls

.. autosummary::
   :toctree: generated/

   RecursiveLSResults

.. currentmodule:: statsmodels.regression.rolling

.. autosummary::
   :toctree: generated/

   RollingRegressionResults

.. currentmodule:: statsmodels.regression.process_regression

.. autosummary::
   :toctree: generated/

   ProcessMLEResults

.. currentmodule:: statsmodels.regression.dimred

.. autosummary::
   :toctree: generated/

   DimReductionResults
