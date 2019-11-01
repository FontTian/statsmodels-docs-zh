.. module:: statsmodels.imputation.mice
   :synopsis: Multiple imputation for missing data

.. currentmodule:: statsmodels.imputation.mice

.. _imputation:


链式方程的多重插补
==========================================

 MICE 模块允许大多数 statsmodels 型拟合独立和/或因变量上具有缺失值的数据集，
 并为拟合参数提供严格的标准误差。基本思想是将具有缺失值的每个变量视为回归中的因变量，
 其中一些或所有剩余变量作为其预测变量。 MICE 程序循环遍历这些模型，依次拟合每个模型，
 然后使用称为 "预测平均匹配" (PMM) 的过程从拟合模型确定的预测分布中生成随机抽取。
 这些随机抽取成为一个插补数据集的估算值。

默认情况下，每个具有缺失变量的变量都使用线性回归建模，拟合数据集中的所有其他变量。
请注意，即使插补模型是线性的， PMM 过程也会保留每个变量的域。因此，例如，
如果给定变量的所有观测值都是正，则变量的所有估算值将始终为正。
用户还可以选择指定使用哪个模型为每个变量生成插补值。

.. code


类
-------

.. currentmodule:: statsmodels.imputation.mice

.. autosummary::
   :toctree: generated/

   MICE
   MICEData

.. currentmodule:: statsmodels.imputation.bayes_mi

.. autosummary::
   :toctree: generated/

   MI
   BayesGaussMI


实施细节
----------------------

在内部，此函数使用 `pandas.isnull <https://pandas.pydata.org/pandas-docs/stable/missing_data.html#working-with-missing-data>`_.
从此函数返回True的任何内容都将被视为丢失的数据。