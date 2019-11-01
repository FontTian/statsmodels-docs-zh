.. currentmodule:: statsmodels.emplike


.. _emplike:


经验似然 :mod:`emplike`
====================================


介绍
------------

Empirical likelihood 经验似然性是一种非参数推理和估计的方法，它消除了必须指定一组基础分布的义务。
此外，经验似然方法不需要重新采样，但仍然唯一地确定其形状反映数据形状的置信区域。本质上，经验似然尝试
将参数方法和非参数方法的优点结合起来，同时限制它们的缺点。经验似然的主要困难是进行推理所需的计算量大的方法。
:mod:`statsmodels.emplike` 试图提供一个用户友好的界面，使最终用户可以有效地进行经验似然分析，而不必担心计算负担。

目前, :mod:`emplike` 提供了进行假设检验和形成描述性统计的置信区间的方法。目前正在开发经验似然估计和回归，
加速故障时间和工具变量模型的推断。

参考文献
^^^^^^^^^^

经验似然的主要参考是::

    Owen, A.B. "Empirical Likelihood." Chapman and Hall, 2001.



例子
--------

.. ipython:: python

  import numpy as np
  import statsmodels.api as sm

  # 生成数据
  x = np.random.standard_normal(50)

  # 启动 EL
  el = sm.emplike.DescStat(x)

  # 平均值的置信区间
  el.ci_mean()

  # test variance is 1
  el.test_var(1)


模块参考
----------------

.. module:: statsmodels.emplike
   :synopsis: Empirical likelihood tools

.. autosummary::
   :toctree: generated/

   descriptive.DescStat
   descriptive.DescStatUV
   descriptive.DescStatMV
