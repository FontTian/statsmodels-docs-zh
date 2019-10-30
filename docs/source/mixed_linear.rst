.. currentmodule:: statsmodels.regression.mixed_linear_model

.. _mixedlmmod:

线性混合效应模型
===========================

线性混合效应模型用于解决相关数据的回归分析。当使用纵向研究和其他研究设计时，
会产生此类数据，在这些研究中，每个主题都有多个观察结果。一下是一些特定的线性混合效果模型是

* *随机截距模型*，其中组中的所有响应都累加了特定于该组的值。

* *随机斜率模型*，其中组中的响应遵循在观察到的协变量中呈线性的（条件）平均轨迹，
并且斜率（可能是截距）随组而变化。

* *方差分量模型*，其中一个或多个分类协变量的水平与分布图相关。
这些随机项根据其协变量值累加确定每个观察的条件均值。

LME的statsmodels实现主要基于组，这意味着对于不同组中的响应，必须独立实现随机效应。在我们的混合模型实现中，
有两种类型的随机效应：（i）具有未知协方差矩阵的随机系数（可能是向量），以及（ii）从常见单变量分布中独立得出
的随机系数。对于（i）和（ii）而言，随机效应通过其具有特定于组的设计矩阵的矩阵/矢量乘积来影响组的条件均值。

如上（i）所示，随机系数的一个简单示例是:

.. math::

   Y_{ij} = \beta_0 + \beta_1X_{ij} + \gamma_{0i} + \gamma_{1i}X_{ij} + \epsilon_{ij}

Here, :math:`Y_{ij}` is the :math:`j^\rm{th}` measured response for subject
:math:`i`, and :math:`X_{ij}` is a covariate for this response.  The
"fixed effects parameters" :math:`\beta_0` and :math:`\beta_1` are
shared by all subjects, and the errors :math:`\epsilon_{ij}` are
independent of everything else, and identically distributed (with mean
zero).  The "random effects parameters" :math:`\gamma_{0i}` and
:math:`\gamma_{1i}` follow a bivariate distribution with mean zero,
described by three parameters: :math:`{\rm var}(\gamma_{0i})`,
:math:`{\rm var}(\gamma_{1i})`, and :math:`{\rm cov}(\gamma_{0i},
\gamma_{1i})`.  There is also a parameter for :math:`{\rm
var}(\epsilon_{ij})`.

如上（ii）所示，方差成分的一个简单示例是:

.. math::

   Y_{ijk} = \beta_0 + \eta_{1i} + \eta_{2j} + \epsilon_{ijk}

Here, :math:`Y_{ijk}` is the :math:`k^\rm{th}` measured response under
conditions :math:`i, j`.  The only "mean structure parameter" is
:math:`\beta_0`.  The :math:`\eta_{1i}` are independent and
identically distributed with zero mean, and variance :math:`\tau_1^2`,
and the :math:`\eta_{2j}` are independent and identically distributed
with zero mean, and variance :math:`\tau_2^2`.

statsmodels MixedLM处理大多数非交叉随机效应模型和某些交叉模型。为了将交叉随机效应包括在模型中，
有必要将整个数据集视为一个组。然后，可以使用模型的方差成分参数来定义具有交叉和非交叉随机效应的
各种组合的模型。

目前，statsmodels LME框架通过 Wald 检验以及系数的置信区间，轮廓似然分析，
似然比检验和AIC进行估计后推断。

例子
--------

.. ipython:: python

  import statsmodels.api as sm
  import statsmodels.formula.api as smf

  data = sm.datasets.get_rdataset("dietox", "geepack").data

  md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
  mdf = md.fit()
  print(mdf.summary())

更多详细示例：

* `Mixed LM <examples/notebooks/generated/mixed_lm_example.html>`__

维基百科上的一些示例:
`MixedLM 的维基笔记 <https://github.com/statsmodels/statsmodels/wiki/Examples#linear-mixed-models>`_



技术文档
-----------------------

数据被分成不相交的组。
第 :math:`i` 组概率模型为:

.. math::

    Y = X\beta + Z\gamma + Q_1\eta_1 + \cdots + Q_k\eta_k + \epsilon

其中

* :math:`n_i` 是第 :math:`i` 组中的观测值
* :math:`Y` 是一个 :math:`n_i` 维的响应变量
* :math:`X` 是一个 :math:`n_i * k_{fe}` 维的固定效应系数矩阵
* :math:`\beta` 是一个 :math:`k_{fe}`-维的固定效应斜率矩阵
* :math:`Z` 是一个 :math:`n_i * k_{re}` 维的随机效应系数矩阵
* :math:`\gamma` 是一个 :math:`k_{re}`-维的随机向量，均值为 0 ，
  协方差矩阵为 :math:`\Psi`; 每组都有独立实现的 gamma 
* :math:`Q_j` 是第 :math:`j` 个方差成分的 :math:`n_i \times q_j` 维的设计矩阵
* :math:`\eta_j` 是一个 :math:`q_j`-维的随机向量，具有 :math:`\tau_j^2`的方差独立且均匀分布的值
* :math:`\epsilon` 是一个 :math:`n_i` 维的服从i.i.d 正态误差的向量 均值为0，方差为 :math:`\sigma^2`;  :math:`\epsilon`
  值的内部和组间相互独立

在机器学习和REML 估计中，
:math:`Y, X, \{Q_j\}` 和 :math:`Z` 必须完全遵守  :math:`\beta`,
:math:`\Psi`, 和 :math:`\sigma^2` 分布，
且 :math:`\gamma`, :math:`\{\eta_j\}` 和 :math:`\epsilon` 是随机的，所以定义了概率模型

边际均值机构为 :math:`E[Y|X,Z] = X*\beta`。  如果仅关注边际均值结构，则GEE是混合模型的不错选择。

符号:

* :math:`cov_{re}` 是随机效应协方差矩阵 (在上面表示为 :math:`\Psi`) 而 :math:`scale` 是标准误差方差。
  每个方差成分都有一个估计方差参数
  :math:`\tau_j^2` 。  对于单个组来说,
  给定 exog 和 endog 的边际协方差矩阵 
  :math:`scale*I + Z * cov_{re} * Z`, 其中 :math:`Z` 是一组随机效应的设计矩阵

参考文献
^^^^^^^^^^

T实现细节的主要参考了:

*   MJ Lindstrom, DM Bates (1988).  *Newton Raphson 和 EM 算法用于线性混合效应模型的重复测量数据*.  Journal of
    美国统计协会杂志. 第 83 卷, 第 404 期, 第 1014-1022 页.

另外，还参考以下最新文档:

* http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

所有的 likelihood（似然）、 gradient（梯度） 和 Hessian 的计算紧跟
Lindstrom 和 Bates.

以下两个文档是从用户角度编写的:

* http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

* http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

.. Class hierarchy: TODO

   General references for this class of models are

模块参考
----------------

.. module:: statsmodels.regression.mixed_linear_model
   :synopsis: Mixed Linear Models


模型类:

.. autosummary::
   :toctree: generated/

   MixedLM

结果类:

.. autosummary::
   :toctree: generated/

   MixedLMResults
