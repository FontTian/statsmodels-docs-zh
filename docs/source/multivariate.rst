.. module:: statsmodels.multivariate
   :synopsis: Models for multivariate data

.. currentmodule:: statsmodels.multivariate

.. _multivariate:


多元统计 :mod:`multivariate`
===========================================

本节包括多元统计中的方法和算法。


主成分分析
----------------------------

.. module:: statsmodels.multivariate.pca
   :synopsis: Principal Component Analaysis

.. currentmodule:: statsmodels.multivariate.pca

.. autosummary::
   :toctree: generated/

   PCA
   pca


因子分析
---------------

.. currentmodule:: statsmodels.multivariate.factor

.. autosummary::
   :toctree: generated/

   Factor
   FactorResults


因子旋转
---------------

.. currentmodule:: statsmodels.multivariate.factor_rotation

.. autosummary::
   :toctree: generated/

   rotate_factors
   target_rotation
   procrustes
   promax


典型相关
---------------------

.. currentmodule:: statsmodels.multivariate.cancorr

.. autosummary::
   :toctree: generated/

   CanCorr


多元方差统计 MANOVA
------

.. currentmodule:: statsmodels.multivariate.manova

.. autosummary::
   :toctree: generated/

   MANOVA


多元 OLS
---------------

`_MultivariateOLS` 是一个功能有限的模型类。目前它支持多变量假设检验，并用作 MANOVA 的后端

.. currentmodule:: statsmodels.multivariate.multivariate_ols

.. autosummary::
   :toctree: generated/

   _MultivariateOLS
   _MultivariateOLSResults
   MultivariateTestResults
