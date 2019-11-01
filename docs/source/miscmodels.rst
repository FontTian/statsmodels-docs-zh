


.. module:: statsmodels.miscmodels
.. currentmodule:: statsmodels.miscmodels


.. _miscmodels:


其他模型 :mod:`miscmodels`
==============================

:mod:`statsmodels.miscmodels` 包含模型类，尚不属于任何其他类别，或者是尚未完善且很可能仍会更改的基本实现。
其中一些模型是作为通用最大似然框架的示例编写的，还有其他一些可能基于一般矩量方法的模型。

已经检查了此类中的模型的基本情况，但与完整实现相比，它们可能更容易遇到数值问题。例如，仅使用通用最大似然框架
添加了 count.Poisson 标准误差基于 Hessian 的数值评估，而离散 Poisson 使用解析梯度和 Hessian ，并且将更加精确，
尤其是当存在很强的多重共线性的情况下。另一方面，通过将 GenericLikelihoodModel 子类化，可以轻松添加新模型，可以
在零膨胀的 Poisson 模型 miscmodels.count 中看到另一个示例。


Count Models :mod:`count`
--------------------------

.. module:: statsmodels.miscmodels.count
.. currentmodule:: statsmodels.miscmodels.count

.. autosummary::
   :toctree: generated/

   PoissonGMLE
   PoissonOffsetGMLE
   PoissonZiGMLE

具有t分布误差的线性模型
--------------------------------------

此类表明只能通过指定对数可能性的方法来定义新模型。所有统计结果信息均从广义似然模型和结果类继承。
通过一个简单示例，统计结果已在 R 中进行了验证。


.. module:: statsmodels.miscmodels.tmodel
.. currentmodule:: statsmodels.miscmodels.tmodel

.. autosummary::
   :toctree: generated/

   TLinearModel




