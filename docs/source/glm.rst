.. currentmodule:: statsmodels.genmod.generalized_linear_model

.. _glm:

广义线性模型
=========================

广义线性模型当前支持使用一参数指数族进行估计。

有关命令和参数，请参考 `Module Reference` 。

例子
--------

.. ipython:: python
   :okwarning:

   # 加载模块和数据集
   import statsmodels.api as sm
   data = sm.datasets.scotland.load(as_pandas=False)
   data.exog = sm.add_constant(data.exog)

   # 实例化之后 gamma 家族模型一般都有 link 函数。
   gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
   gamma_results = gamma_model.fit()
   print(gamma_results.summary())

更多详细示例:

* `GLM <examples/notebooks/generated/glm.html>`__
* `Formula <examples/notebooks/generated/glm_formula.html>`__

技术文档
-----------------------

..   ..glm_techn1
..   ..glm_techn2

每个观测值的统计模型， :math:`i` 的表达式

 :math:`Y_i \sim F_{EDM}(\cdot|\theta,\phi,w_i)` and
 :math:`\mu_i = E[Y_i|x_i] = g^{-1}(x_i^\prime\beta)`.

当 :math:`g` 是 link 函数， :math:`F_{EDM}(\cdot|\theta,\phi,w)`
is a distribution of the family of exponential dispersion models (EDM) with
natural parameter :math:`\theta`, scale parameter :math:`\phi` and weight
:math:`w`.
Its density is given by

 :math:`f_{EDM}(y|\theta,\phi,w) = c(y,\phi,w)
 \exp\left(\frac{y\theta-b(\theta)}{\phi}w\right)\,.`

It follows that :math:`\mu = b'(\theta)` and
:math:`Var[Y|x]=\frac{\phi}{w}b''(\theta)`. The inverse of the first equation
gives the natural parameter as a function of the expected value
:math:`\theta(\mu)` such that

 :math:`Var[Y_i|x_i] = \frac{\phi}{w_i} v(\mu_i)`

with :math:`v(\mu) = b''(\theta(\mu))`. Therefore it is said that a GLM is
determined by link function :math:`g` and variance function :math:`v(\mu)`
alone (and :math:`x` of course).

Note that while :math:`\phi` is the same for every observation :math:`y_i`
and therefore does not influence the estimation of :math:`\beta`,
the weights :math:`w_i` might be different for every :math:`y_i` such that the
estimation of :math:`\beta` depends on them.

================================================= ============================== ============================== ======================================== =========================================== ============================================================================ =====================
分布                                      域                        :math:`\mu=E[Y|x]`             :math:`v(\mu)`                           :math:`\theta(\mu)`                         :math:`b(\theta)`                                                            :math:`\phi`
================================================= ============================== ============================== ======================================== =========================================== ============================================================================ =====================
二项式分布 :math:`B(n,p)`                           :math:`0,1,\ldots,n`           :math:`np`                     :math:`\mu-\frac{\mu^2}{n}`              :math:`\log\frac{p}{1-p}`                   :math:`n\log(1+e^\theta)`                                                    1
泊松分布 :math:`P(\mu)`                            :math:`0,1,\ldots,\infty`      :math:`\mu`                    :math:`\mu`                              :math:`\log(\mu)`                           :math:`e^\theta`                                                             1
负二项分布 :math:`NB(\mu,\alpha)`                :math:`0,1,\ldots,\infty`      :math:`\mu`                    :math:`\mu+\alpha\mu^2`                  :math:`\log(\frac{\alpha\mu}{1+\alpha\mu})` :math:`-\frac{1}{\alpha}\log(1-\alpha e^\theta)`                             1
高斯/标准正太分布 :math:`N(\mu,\sigma^2)`           :math:`(-\infty,\infty)`       :math:`\mu`                    :math:`1`                                :math:`\mu`                                 :math:`\frac{1}{2}\theta^2`                                                  :math:`\sigma^2`
Gamma分布 :math:`N(\mu,\nu)`                          :math:`(0,\infty)`             :math:`\mu`                    :math:`\mu^2`                            :math:`-\frac{1}{\mu}`                      :math:`-\log(-\theta)`                                                       :math:`\frac{1}{\nu}`
逆高斯分布. :math:`IG(\mu,\sigma^2)`              :math:`(0,\infty)`             :math:`\mu`                    :math:`\mu^3`                            :math:`-\frac{1}{2\mu^2}`                   :math:`-\sqrt{-2\theta}`                                                     :math:`\sigma^2`
Tweedie分布 :math:`p\geq 1`                           depends on :math:`p`           :math:`\mu`                    :math:`\mu^p`                            :math:`\frac{\mu^{1-p}}{1-p}`               :math:`\frac{\alpha-1}{\alpha}\left(\frac{\theta}{\alpha-1}\right)^{\alpha}` :math:`\phi`
================================================= ============================== ============================== ======================================== =========================================== ============================================================================ =====================

Tweedie分布具有 :math:`p=0,1,2` 的特殊情况
没有在表中列出和使用 :math:`\alpha=\frac{p-2}{p-1}`.

数学变量与代码的对应关系:

* :math:`Y` 和 :math:`y` 编码为 ``endog``, 要对变量 math:`Y` 建模
* :math:`x` 编码为 ``exog``, 协变量别名解释变量
* :math:`\beta` 编码为 ``params``, 要估算的参数
* :math:`\mu` 编码为 ``mu``, (基于 :math:`x`) math:`Y`的期望。
* :math:`g` 编码为 ``link`` 传递给 ``class Family``
* :math:`\phi` 编码为 ``scale``, EDM 的参数
* :math:`w` 目前不支持 (i.e. :math:`w=1`)的情况, 未来有可能支持
  ``var_weights``
* :math:`p` 编码为 ``var_power`` 方差函数的幂服从
  :math:`v(\mu)` 分布, 请参见表
* :math:`\alpha` 或者是

  * 负二项式: 辅助参数 ``alpha``, 请参见表
  * Tweedie: Tweedie 的缩写 :math:`\frac{p-2}{p-1}`  :math:`p`
    的函数, 请参见表


参考文献
^^^^^^^^^^

* Gill, Jeff. 2000. 广义线性模型：统一方法. SAGE QASS 系列.
* Green, PJ. 1984. “迭代加权最小二乘以求最大似然估计，以及一些健壮和可靠的替代方案。”皇家统计学会杂志， B 系列, 46, 149-192.
* Hardin, J.W. and Hilbe, J.M. 2007. “广义线性模型和扩展”，第二版. Stata Press, College Station, TX.
* McCullagh, P. and Nelder, J.A. 1989. “广义线性模型”。第二版. Chapman & Hall, Boca Rotan.

模块参考
----------------

.. module:: statsmodels.genmod.generalized_linear_model
   :synopsis: 广义线性模型 (GLM)

模型类
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLM

结果类
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLMResults
   PredictionResults

.. _families:

家族模型
^^^^^^^^

当前发行的家族模型有

.. module:: statsmodels.genmod.families.family
.. currentmodule:: statsmodels.genmod.families.family

.. autosummary::
   :toctree: generated/

   Family
   Binomial
   Gamma
   Gaussian
   InverseGaussian
   NegativeBinomial
   Poisson
   Tweedie


.. _links:

Link 函数
^^^^^^^^^^^^^^

下表是当前实现了 link 函数，并不是每个家族模型都可以使用 link 函数。
可以使用 link 函数的如下表


::

    >>> sm.families.family.<familyname>.links

.. module:: statsmodels.genmod.families.links
.. currentmodule:: statsmodels.genmod.families.links

.. autosummary::
   :toctree: generated/

   Link
   CDFLink
   CLogLog
   Log
   Logit
   NegativeBinomial
   Power
   cauchy
   cloglog
   identity
   inverse_power
   inverse_squared
   log
   logit
   nbinom
   probit

.. _varfuncs:

方差函数
^^^^^^^^^^^^^^^^^^

每个族都有关联的方差函数。方差函数如下表：

::

    >>> sm.families.<familyname>.variance

.. module:: statsmodels.genmod.families.varfuncs
.. currentmodule:: statsmodels.genmod.families.varfuncs

.. autosummary::
   :toctree: generated/

   VarianceFunction
   constant
   Power
   mu
   mu_squared
   mu_cubed
   Binomial
   binary
   NegativeBinomial
   nbinom
