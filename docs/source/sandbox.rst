.. _sandbox:


Sandbox
=======

This sandbox 包含出于各种原因尚未准备好包含在 statsmodels 中的代码，它包含未经测试，验证和更新为新 statsmodels 结构的
旧 stats.models 代码模块，cox 生存模型, 具有重复度量的混合效应模型，广义加性模型和公式框架。 sandbox 还包含当前正在处理
的代码，直到它适合 statsmodels 模块或经过充分测试为止。

必须明确导入所有 sandbox 模块，以指示它们还不是 statsmodels 核心的一部分。沙盒代码的质量和测试差异很大。


例子
--------

 `sandbox.examples`文件夹中有一些示例。其他示例直接包含在 sandbox 的模块和子文件夹中。


模块参考
----------------


时间序列分析 :mod:`tsa`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在这一部分中，我们将开发对时间序列分析有用的模型和函数。大多数模型和功能已移至 :mod:`statsmodels.tsa` 。


Moving Window Statistics
""""""""""""""""""""""""

Most moving window statistics, like rolling mean, moments (up to 4th order), min,
max, mean, and variance, are covered by the functions for `Moving (rolling)
statistics/moments <https://pandas.pydata.org/pandas-docs/stable/computation.html#moving-rolling-statistics-moments>`_ in Pandas.

.. module:: statsmodels.sandbox.tsa
   :synopsis: Experimental time-series analysis models

.. currentmodule:: statsmodels.sandbox.tsa

.. autosummary::
   :toctree: generated/

   movstat.movorder
   movstat.movmean
   movstat.movvar
   movstat.movmoment


回归与方差分析
^^^^^^^^^^^^^^^^^^^^

.. module:: statsmodels.sandbox.regression.anova_nistcertified
   :synopsis: 实验方差分析估算器

.. currentmodule:: statsmodels.sandbox.regression.anova_nistcertified

以下两个 ANOVA 函数已针对平衡单向 ANOVA 的 NIST 测试数据进行了全面检验。anova_oneway 遵循与 scipy.stats 中
的单向方差函数相同的模式，但对于严重缩放的问题具有更高的精度。 ``anova_ols`` 产生相同的结果作为然而使用 OLS 
模型类的单向 ANOVA。它还可以针对 NIST 测试进行验证，在最坏的情况下会出现一些问题。它显示了如何在三行中使用 
statsmodels进行简单的 ANOVA 并且最好将其作为配方。

.. autosummary::
   :toctree: generated/

   anova_oneway
   anova_ols


以下是用于处理伪变量并使用 OLS 生成 ANOVA 结果的辅助函数。最好将它们视为食谱，
因为它们在编写时会考虑到特定的用途。这些功能最终将被重写或重组。

.. module:: statsmodels.sandbox.regression
   :synopsis: 实验回归工具

.. currentmodule:: statsmodels.sandbox.regression

.. autosummary::
   :toctree: generated/

   try_ols_anova.data2dummy
   try_ols_anova.data2groupcont
   try_ols_anova.data2proddummy
   try_ols_anova.dropname
   try_ols_anova.form2design

以下是用于组统计的帮助程序功能，其中组由标签数组定义。前一组的合格注释也适用于该组功能。


.. autosummary::
   :toctree: generated/

   try_catdata.cat2dummy
   try_catdata.convertlabels
   try_catdata.groupsstats_1d
   try_catdata.groupsstats_dummy
   try_catdata.groupstatsbin
   try_catdata.labelmeanfilter
   try_catdata.labelmeanfilter_nd
   try_catdata.labelmeanfilter_str

除了这些功能之外，sandbox 回归还包含几个示例，这些示例说明了 statsmodels 回归模型的使用。



回归方程组和联立方程组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下是方程模型的拟合系统。尽管返回的参数已经过验证，但该代码仍处于实验阶段，
在将模型添加到主代码库之前，它们的用法很有可能会发生重大变化。

.. module:: statsmodels.sandbox.sysreg
   :synopsis: Experimental system regression models

.. currentmodule:: statsmodels.sandbox.sysreg

.. autosummary::
   :toctree: generated/

   SUR
   Sem2SLS

Miscellaneous
^^^^^^^^^^^^^
.. module:: statsmodels.sandbox.tools.tools_tsa
   :synopsis: Experimental tools for working with time-series

.. currentmodule:: statsmodels.sandbox.tools.tools_tsa


描述性统计输出
"""""""""""""""""""""""""""""""

.. module:: statsmodels.sandbox
   :synopsis: Experimental tools that have not been fully vetted

.. currentmodule:: statsmodels.sandbox

.. autosummary::
   :toctree: generated/

   descstats.sign_test
   descstats.descstats




原始的 stats.models
^^^^^^^^^^^^^^^^^^^^^

None of these are fully working. The formula framework is used by cox and
mixed.

**Mixed Effects Model with Repeated Measures using an EM Algorithm**

:mod:`statsmodels.sandbox.mixed`


**Cox Proportional Hazards Model**

:mod:`statsmodels.sandbox.cox`

**Generalized Additive Models**

:mod:`statsmodels.sandbox.gam`

**Formula**

:mod:`statsmodels.sandbox.formula`


