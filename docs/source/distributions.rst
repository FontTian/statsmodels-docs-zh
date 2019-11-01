.. module:: statsmodels.sandbox.distributions
   :synopsis: Probability distributions

.. currentmodule:: statsmodels.sandbox.distributions

.. _distributions:


分布
=============

本节收集用于统计分布的各种其他功能和方法。

经验分布
-----------------------

.. module:: statsmodels.distributions.empirical_distribution
   :synopsis: 用于经验分布的工具

.. currentmodule:: statsmodels.distributions.empirical_distribution

.. autosummary::
   :toctree: generated/

   ECDF
   StepFunction
   monotone_fn_inverter

Distribution Extras
-------------------


.. module:: statsmodels.sandbox.distributions.extras
   :synopsis: 概率分布和随机数生成器

.. currentmodule:: statsmodels.sandbox.distributions.extras

*偏斜分布*

.. autosummary::
   :toctree: generated/

   SkewNorm_gen
   SkewNorm2_gen
   ACSkewT_gen
   skewnorm2

*基于 Gram-Charlier 展开的分布*

.. autosummary::
   :toctree: generated/

   pdf_moments_st
   pdf_mvsk
   pdf_moments
   NormExpan_gen

scipy.stats 的 *多元正态分布* 封装 


.. autosummary::
   :toctree: generated/

   mvstdnormcdf
   mvnormcdf

非线性变换的单变量分布
------------------------------------------------------

可以从现有单变量分布生成一个非线性变换的单变量分布， `Transf_gen` 类是可以从单调变换生成新分布的类
`TransfTwo_gen` 可以使用 驼峰形 或 u形 变换，例如  abs 或 square. 其余对象是特殊情况。

.. module:: statsmodels.sandbox.distributions.transformed
   :synopsis: 实验概率分布和随机数生成器

.. currentmodule:: statsmodels.sandbox.distributions.transformed

.. autosummary::
   :toctree: generated/

   TransfTwo_gen
   Transf_gen

   ExpTransf_gen
   LogTransf_gen
   SquareFunc

   absnormalg
   invdnormalg

   loggammaexpg
   lognormalg
   negsquarenormalg

   squarenormalg
   squaretg
