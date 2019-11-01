.. currentmodule:: statsmodels.tools


.. _tools:

工具
=====

我们的工具集包含一些方便用户的功能，以及主要供内部使用而编写的功能。

除了此工具目录外，其他几个子程序包也有自己的工具模块，例如 :mod:`statsmodels.tsa.tsatools`


模块参考
----------------

.. module:: statsmodels.tools
   :synopsis: 用于变量转换和常用数值运算的工具

基本工具 :mod:`tools`
^^^^^^^^^^^^^^^^^^^^^^^^

这些是基本的杂项工具。完整的导入路径是 `statsmodels.tools.tools` 。

.. autosummary::
   :toctree: generated/

   tools.add_constant

下一组主要是未单独测试或未充分测试的辅助功能。

.. autosummary::
   :toctree: generated/

   tools.categorical
   tools.clean0
   tools.fullrank
   tools.isestimable
   tools.recipr
   tools.recipr0
   tools.unsqueeze

.. currentmodule:: statsmodels.tools

.. _numdiff:

数值微分
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   numdiff.approx_fprime
   numdiff.approx_fprime_cs
   numdiff.approx_hess1
   numdiff.approx_hess2
   numdiff.approx_hess3
   numdiff.approx_hess_cs

衡量拟合效果 :mod:`eval_measures`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此模块中的第一组功能是信息标准的独立版本 aic bic 和 hqic. 带 `_sigma` 后缀的函数将误差平方和作为参数，
没有后缀的函数将对数似然值 `llf` 作为参数。

第二组功能是衡量拟合或预测的效果,它们通常是用作辅助功能，所有这些都是为了衡量两个数组之间的效果或统计信息差异。
例如，在 Monte Carlo（蒙特卡洛）或交叉验证的情况下，第一个数组将是不同重复或绘图的估计结果，而第二个数组将是真实值或观察值。

.. currentmodule:: statsmodels.tools

.. autosummary::
   :toctree: generated/

   eval_measures.aic
   eval_measures.aic_sigma
   eval_measures.aicc
   eval_measures.aicc_sigma
   eval_measures.bic
   eval_measures.bic_sigma
   eval_measures.hqic
   eval_measures.hqic_sigma

   eval_measures.bias
   eval_measures.iqr
   eval_measures.maxabs
   eval_measures.meanabs
   eval_measures.medianabs
   eval_measures.medianbias
   eval_measures.mse
   eval_measures.rmse
   eval_measures.stde
   eval_measures.vare
