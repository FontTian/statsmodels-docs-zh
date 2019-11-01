.. currentmodule:: statsmodels.discrete.discrete_model


.. _discretemod:

离散因变量回归
===========================================

有限和定性因变量的回归模型。该模块当前允许使用二值化 (Logit, Probit), 名义
(MNLogit), 或计数 (Poisson, NegativeBinomial) 数据来估计模型。

从0.9版开始，这还包括新的计数模型，这些模型仍在0.9中进行试验 NegativeBinomialP, GeneralizedPoisson 和 zero-inflated
models, ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP 和 ZeroInflatedGeneralizedPoisson.

有关命令和参数，请参见 `Module Reference`_ 。

例子
--------

.. ipython:: python
  :okwarning:

  # 加载数据 Spector 和 Mazzeo (1980)
  import statsmodels.api as sm
  spector_data = sm.datasets.spector.load_pandas()
  spector_data.exog = sm.add_constant(spector_data.exog)

  # Logit 模型
  logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
  logit_res = logit_mod.fit()
  print(logit_res.summary())

更多详细示例:


* `总览 <examples/notebooks/generated/discrete_choice_overview.html>`__
* `例子 <examples/notebooks/generated/discrete_choice_example.html>`__

技术文档
-----------------------

当前，所有模型都是通过最大似然估计的，并假设独立且均等分布的误差。

所有离散回归模型都定义相同的方法并遵循相同的结构，这与回归结果相似，但其中有离散模型一些特定方法。此外，其中还包含其他特定于模型的方法和属性。


参考文献
^^^^^^^^^^

此类模型的一般参考是::

    A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
        Cambridge, 1998

    G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
        Cambridge, 1983.

    W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.

模块参考
----------------

.. module:: statsmodels.discrete.discrete_model
   :synopsis: 离散数据模型

特定的模型类是:

.. autosummary::
   :toctree: generated/

   Logit
   Probit
   MNLogit
   Poisson
   NegativeBinomial
   NegativeBinomialP
   GeneralizedPoisson

.. currentmodule:: statsmodels.discrete.count_model
.. module:: statsmodels.discrete.count_model

.. autosummary::
   :toctree: generated/

   ZeroInflatedPoisson
   ZeroInflatedNegativeBinomialP
   ZeroInflatedGeneralizedPoisson

.. currentmodule:: statsmodels.discrete.conditional_models
.. module:: statsmodels.discrete.conditional_models

.. autosummary::
   :toctree: generated/

   ConditionalLogit
   ConditionalMNLogit
   ConditionalPoisson

具体的结果类是:

.. currentmodule:: statsmodels.discrete.discrete_model

.. autosummary::
   :toctree: generated/

   LogitResults
   ProbitResults
   CountResults
   MultinomialResults
   NegativeBinomialResults
   GeneralizedPoissonResults

.. currentmodule:: statsmodels.discrete.count_model

.. autosummary::
   :toctree: generated/

   ZeroInflatedPoissonResults
   ZeroInflatedNegativeBinomialResults
   ZeroInflatedGeneralizedPoissonResults


:class:`DiscreteModel` 是所有离散回归模型的超类。估算结果作为 :class:`DiscreteResults`子类之一的实例返回 。
模型的每个类别（ binary, count 和 multinomial）都有自己的中间级别的模型和结果类。这中间阶级大多是便于通过定义
的方法和属性的实现 :class:`DiscreteModel` 和 :class:`DiscreteResults` 。

.. currentmodule:: statsmodels.discrete.discrete_model

.. autosummary::
   :toctree: generated/

   DiscreteModel
   DiscreteResults
   BinaryModel
   BinaryResults
   CountModel
   MultinomialModel

.. currentmodule:: statsmodels.discrete.count_model

.. autosummary::
   :toctree: generated/

   GenericZeroInflated
