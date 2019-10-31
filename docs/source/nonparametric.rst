.. currentmodule:: statsmodels.nonparametric

.. _nonparametric:


非参数方法 :mod:`nonparametric`
==========================================

本节收集非参数统计中的各种方法。这包括用于单变量和多变量数据的核密度估计，
核回归和局部加权散点图平滑 (lowess).

sandbox.nonparametric 包含正在进行的工作或尚未进行单元测试的其他功能。
我们计划在此包括非参数密度估计量，特别是基于核或正交多项式，平滑器以及
用于统计模型其他部分的非参数模型和方法的工具。


内核密度估计
-------------------------

内核密度估计 (KDE) 功能分为单变量估计和多变量估计，这两种实现方式完全不同。

单变量估计 (由 `KDEUnivariate` 提供) 使用FFT变换，这使其速度非常快。
因此，如果速度很重要，则对于连续的单变量数据应该是首选。它支持使用不同的内核。
带宽估计只能通过经验法则(Scott or Silverman).

多变量估计 (由 `KDEMultivariate` 提供) u使用乘积核。它支持最小二乘和最大似然交叉验证，
以进行带宽估计，以及估计混合的连续、有序和无序数据。但是，目前无法更改默认内核 (Gaussian, Wang-Ryzin 和
Aitchison-Aitken) 。 `KDEMultivariateConditional` 支持对条件密度 (:math:`P(X | Y) = P(X, Y) / P(Y)`) 的直接估计。

`KDEMultivariate` 也可以进行单变量估计，但比 `KDEUnivariate` 慢两个数量级


Kernel 回归
-----------------

Kernel 回归 (由 `KernelReg` 提供) 基于与 `KDEMultivariate` 相同的乘积内核方法，
因此具有与上述 `KDEMultivariate` 相同的功能集（混合数据，交叉验证的带宽估计，内核），
审查的回归由 `KernelCensoredReg` 提供。

请注意，可以在沙箱中找到基于 `KernelReg` 的半参数部分线性模型和单索引模型的代码。

参考文献
----------

* B.W. Silverman, "Density Estimation for Statistics and Data Analysis"
* J.S. Racine, "Nonparametric Econometrics: A Primer," Foundation and
  Trends in Econometrics, Vol. 3, No. 1, pp. 1-88, 2008.
* Q. Li and J.S. Racine, "Nonparametric econometrics: theory and practice",
  Princeton University Press, 2006.
* Hastie, Tibshirani and Friedman, "The Elements of Statistical Learning:
  Data Mining, Inference, and Prediction", Springer, 2009.
* Racine, J., Li, Q. "Nonparametric Estimation of Distributions
  with Categorical and Continuous Data." Working Paper. (2000)
* Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
  Distributions Annals of Economics and Finance 5, 211-235 (2004)
* Liu, R., Yang, L. "Kernel estimation of multivariate
  cumulative distribution function." Journal of Nonparametric Statistics 
  (2008)
* Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
  with Categorical and Continuous Data." Working Paper
* Li, Q., Racine, J. "Cross-validated local linear nonparametric
  regression" Statistica Sinica 14(2004), pp. 485-512
* Racine, J.: "Consistent Significance Testing for Nonparametric
  Regression" Journal of Business & Economics Statistics
* Racine, J., Hart, J., Li, Q., "Testing the Significance of
  Categorical Predictor Variables in Nonparametric Regression
  Models", 2006, Econometric Reviews 25, 523-544


模块参考
----------------

.. module:: statsmodels.nonparametric
   :synopsis: 密度和曲线的非参数估计

公开的功能和类是

.. currentmodule:: statsmodels.nonparametric.smoothers_lowess
.. autosummary::
   :toctree: generated/

   lowess

.. currentmodule:: statsmodels.nonparametric.kde
.. autosummary::
   :toctree: generated/

   KDEUnivariate

.. currentmodule:: statsmodels.nonparametric.kernel_density
.. autosummary::
   :toctree: generated/

   KDEMultivariate
   KDEMultivariateConditional
   EstimatorSettings

.. currentmodule:: statsmodels.nonparametric.kernel_regression
.. autosummary::
   :toctree: generated/

   KernelReg
   KernelCensoredReg

内核带宽的辅助函数

.. currentmodule:: statsmodels.nonparametric.bandwidths
.. autosummary::
   :toctree: generated/

   bw_scott
   bw_silverman
   select_bandwidth

:mod:`statsmodels.nonparametric.dgp_examples` 有一些非线性函数的示例。

 sandbox.nonparametric 包含其他未充分检验的类，这些类用于测试功能形式以及半线性和单索引模型。
