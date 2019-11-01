:orphan:

.. _diagnostics:

回归诊断和设定性检验
==============================================


介绍
------------

在许多统计分析情况下，我们不确定统计模型是否正确指定，例如，当使用ols时，假定线性和均方差，
另外一些测试统计数据还假定误差是正态分布的，或者我们有大量样本。由于我们的结果取决于这些统计假设，
因此结果仅在我们的假设成立（至少近似）时才是正确的。

正确地解决设定性问题的一种方法是使用鲁棒性的方法，例如，鲁棒性回归或鲁棒性（sandwich）协方差。
第二种方法是测试我们的样本是否符合这些假设。

以下简要概述了线性回归的回归诊断和假设检验

异方差测试
------------------------

对于这些检验，零假设是所有观测值都具有相同的误差方差，即误差是同调的。检验的不同之处在于，
哪种异方差被视为替代假设。对于不同类型的异方差性，它们的测试能力也会有所不同。

:py:func:`het_breuschpagan <statsmodels.stats.diagnostic.het_breuschpagan>`
    Lagrange Multiplier Heteroscedasticity Test by Breusch-Pagan

:py:func:`het_white <statsmodels.stats.diagnostic.het_white>`
    Lagrange Multiplier Heteroscedasticity Test by White

:py:func:`het_goldfeldquandt <statsmodels.stats.diagnostic.het_goldfeldquandt>`
    test whether variance is the same in 2 subsamples


自相关检验
---------------------

这组测试回归残差是否自相关的。他们假定观测值是按时间排序的。

:py:func:`durbin_watson <statsmodels.stats.diagnostic.durbin_watson>`
  - Durbin-Watson test for no autocorrelation of residuals
  - printed with summary()

:py:func:`acorr_ljungbox <statsmodels.stats.diagnostic.acorr_ljungbox>`
  - Ljung-Box test for no autocorrelation of residuals
  - also returns Box-Pierce statistic

:py:func:`acorr_breusch_godfrey <statsmodels.stats.diagnostic.acorr_breusch_godfrey>`
  - Breusch-Pagan test for no autocorrelation of residuals


missing
  - ?


非线性检验
-------------------

:py:func:`linear_harvey_collier <statsmodels.stats.diagnostic.linear_harvey_collier>`
  - Multiplier test for Null hypothesis that linear specification is
    correct

:py:func:`acorr_linear_rainbow <statsmodels.stats.diagnostic.acorr_linear_rainbow>`
  - Multiplier test for Null hypothesis that linear specification is
    correct.

:py:func:`acorr_linear_lm <statsmodels.stats.diagnostic.acorr_linear_lm>`
  - Lagrange Multiplier test for Null hypothesis that linear specification is
    correct. This tests against specific functional alternatives.

:py:func:`spec_white <statsmodels.stats.diagnostic.spec_white>`
  - White's two-moment specification test with null hypothesis of homoscedastic
    and correctly specified.

结构变化、参数稳定性检验
------------------------------------------------

检验全部或某些回归系数在整个数据样本中是否恒定。

已知变更点
^^^^^^^^^^^^^^^^^^

OneWayLS :
  - flexible ols wrapper for testing identical regression coefficients across
    predefined subsamples (eg. groups)

missing
  - predictive test: Greene, number of observations in subsample is smaller than
    number of regressors


未知变更点
^^^^^^^^^^^^^^^^^^^^

:py:func:`breaks_cusumolsresid <statsmodels.stats.diagnostic.breaks_cusumolsresid>`
  - cusum test for parameter stability based on ols residuals

:py:func:`breaks_hansen <statsmodels.stats.diagnostic.breaks_hansen>`
  - test for model stability, breaks in parameters for ols, Hansen 1992

:py:func:`recursive_olsresiduals <statsmodels.stats.diagnostic.recursive_olsresiduals>`
  Calculate recursive ols with residuals and cusum test statistic. This is
  currently mainly helper function for recursive residual based tests.
  However, since it uses recursive updating and does not estimate separate
  problems it should be also quite efficient as expanding OLS function.

missing
  - supLM, expLM, aveLM  (Andrews, Andrews/Ploberger)
  - R-structchange also has musum (moving cumulative sum tests)
  - test on recursive parameter estimates, which are there?


多重共线性检验
--------------------------------

conditionnum (statsmodels.stattools)
  - -- needs test vs Stata --
  - cf Grene (3rd ed.) pp 57-8

numpy.linalg.cond
  - (for more general condition numbers, but no behind the scenes help for
    design preparation)

方差膨胀因素
  This is currently together with influence and outlier measures
  (with some links to other tests here: http://www.stata.com/help.cgi?vif)


正态分布检验
--------------------------------

:py:func:`jarque_bera <statsmodels.stats.tools.jarque_bera>`
  - printed with summary()
  - test for normal distribution of residuals

科学统计中的正态性检验
  need to find list again

:py:func:`omni_normtest <statsmodels.stats.tools.omni_normtest>`
  - 检验残差的正态分布
  - printed with summary()

:py:func:`normal_ad <statsmodels.stats.diagnostic.normal_ad>`
  - Anderson Darling 检验均值和方差的正态性
:py:func:`kstest_normal <statsmodels.stats.diagnostic.kstest_normal>` :py:func:`lilliefors <statsmodels.stats.diagnostic.lilliefors>`
  Lilliefors test for normality, this is a Kolmogorov-Smirnov tes with for
  normality with estimated mean and variance. lilliefors is an alias for
  kstest_normal

qqplot, scipy.stats.probplot

other goodness-of-fit tests for distributions in scipy.stats and enhancements
  - kolmogorov-smirnov
  - anderson : Anderson-Darling
  - likelihood-ratio, ...
  - chisquare tests, powerdiscrepancy : needs wrapping (for binning)


异常值和影响的诊断措施
-----------------------------------------

这些措施试图确定离群值较大，残差较大的观测值或对回归估计值影响较大的观测值。稳健回归RLM
可用于以异常健壮的方式进行估计以及识别异常。RLM的优点是，即使存在许多异常值，估计结果也
不会受到很大的影响，而大多数其他措施则可以更好地识别单个异常值，并且可能无法识别异常值组。

:py:class:`RLM <statsmodels.robust.robust_linear_model.RLM>`
    示例来自 example_rlm.py ::

        import statsmodels.api as sm

        ### Example for using Huber's T norm with the default
        ### median absolute deviation scaling

        data = sm.datasets.stackloss.load()
        data.exog = sm.add_constant(data.exog)
        huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
        hub_results = huber_t.fit()
        print(hub_results.weights)

    权重给出了根据要求的缩放比例将特定观察权降低多少的想法。
  

:py:class:`Influence <statsmodels.stats.outliers_influence.OLSInfluence>`
   stats.outliers_influence 中的类, 对于离群值和影响力的大多数标准度量都可以作为给定的 OLS 模型提供的方法或属性来使用。 
   这主要是为OLS编写的，某些（但不是全部）量度对其他模型也有效。这些统计信息中的一些可以从OLS结果实例计算得出，其他一些则需要为每个遗漏变量估算OLS。

   - resid_press
   - resid_studentized_external
   - resid_studentized_internal
   - ess_press
   - hat_matrix_diag
   - cooks_distance - Cook's Distance `Wikipedia <https://en.wikipedia.org/wiki/Cook%27s_distance>`_ (with some other links)
   - cov_ratio
   - dfbetas
   - dffits
   - dffits_internal
   - det_cov_params_not_obsi
   - params_not_obsi
   - sigma2_not_obsi



单位根检验
---------------

:py:func:`unitroot_adf <statsmodels.stats.diagnostic.unitroot_adf>`
  - same as adfuller but with different signature
