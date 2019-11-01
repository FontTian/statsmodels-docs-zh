.. currentmodule:: statsmodels.sandbox.regression.gmm


.. _gmm:


广义矩量法 :mod:`gmm`
========================================

:mod:`statsmodels.gmm` 包含基于通用矩量估计的模型类和函数。当前，一般的非线性情况已经实现。
包含标准线性工具变量模型的示例类。这是作为一个测试用例引入的，它可以正常工作，但是没有考虑线性结构。
对于线性情况，我们打算引入一个特定的实现，该实现将更快并且在数值上更加准确。

当前, GMM 采用任意非线性矩条件，并通过给定最佳加权矩阵和估计参数之间交替来计算给定加权矩阵的估计值
或迭代计算。通过将 GMM 子类化，可以实现具有不同力矩条件的模型。在最小的实现中，仅需定义即时条件 `momcond` 。

.. currentmodule:: statsmodels.sandbox.regression.gmm


模块参考
""""""""""""""""

.. module:: statsmodels.sandbox.regression.gmm
   :synopsis: A framework for implementing Generalized Method of Moments (GMM)

.. autosummary::
   :toctree: generated/

   GMM
   GMMResults
   IV2SLS
   IVGMM
   IVGMMResults
   IVRegressionResults
   LinearIVGMM
   NonlinearIVGMM
