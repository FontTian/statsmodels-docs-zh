.. module:: statsmodels.stats
   :synopsis: Statistical methods and tests

.. currentmodule:: statsmodels.stats

.. _stats:


统计 :mod:`stats`
=======================

本节收集各种统计测试和工具。某些模型可以独立于任何模型使用，而某些模型则是对模型和模型结果的扩展。

API警告：此类别中的函数和对象分布在各个模块中，并且可能仍在移动。我们希望将来统计检验将返回具有更多信息报告的类实例，而不仅仅是原始数字。


.. _stattools:


Residual Diagnostics and Specification Tests
--------------------------------------------

.. module:: statsmodels.stats.stattools
   :synopsis: Statistical methods and tests that do not fit into other categories

.. currentmodule:: statsmodels.stats.stattools

.. autosummary::
   :toctree: generated/

   durbin_watson
   jarque_bera
   omni_normtest
   medcouple
   robust_skewness
   robust_kurtosis
   expected_robust_kurtosis

.. module:: statsmodels.stats.diagnostic
   :synopsis: 用于诊断模型拟合问题的统计方法和检验

.. currentmodule:: statsmodels.stats.diagnostic

.. autosummary::
   :toctree: generated/

   acorr_ljungbox
   acorr_breusch_godfrey

   HetGoldfeldQuandt
   het_goldfeldquandt
   het_breuschpagan
   het_white
   het_arch

   linear_harvey_collier
   linear_rainbow
   linear_lm

   breaks_cusumolsresid
   breaks_hansen
   recursive_olsresiduals

   CompareCox
   compare_cox
   CompareJ
   compare_j

   unitroot_adf

   normal_ad
   kstest_normal
   lilliefors

异常值和影响措施
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: statsmodels.stats.outliers_influence
   :synopsis: 异常值和影响的统计方法和度量

.. currentmodule:: statsmodels.stats.outliers_influence

.. autosummary::
   :toctree: generated/

   OLSInfluence
   GLMInfluence
   MLEInfluence
   variance_inflation_factor

另请参阅 ::`关于回归诊断的注释： <diagnostics>` 。

Sandwich Robust 协方差
---------------------------

以下函数针对参数估计计算协方差矩阵和标准误差，这些参数估计对误差的异方差性和自相关具有鲁棒性。
与 LinearModelResults 可用的方法类似，这些方法设计用于 OLS。

.. currentmodule:: statsmodels.stats

.. autosummary::
   :toctree: generated/

   sandwich_covariance.cov_hac
   sandwich_covariance.cov_nw_panel
   sandwich_covariance.cov_nw_groupsum
   sandwich_covariance.cov_cluster
   sandwich_covariance.cov_cluster_2groups
   sandwich_covariance.cov_white_simple

以下是 LinearModelResults 附带的异方差稳健性标准误差的独立版本

.. autosummary::
   :toctree: generated/

   sandwich_covariance.cov_hc0
   sandwich_covariance.cov_hc1
   sandwich_covariance.cov_hc2
   sandwich_covariance.cov_hc3

   sandwich_covariance.se_cov


拟合优度和量度
----------------------------------

一些关于单变量分布拟合优度的检验

.. module:: statsmodels.stats.gof
   :synopsis: 拟合度和检验的优度

.. currentmodule:: statsmodels.stats.gof

.. autosummary::
   :toctree: generated/

   powerdiscrepancy
   gof_chisquare_discrete
   gof_binning_discrete
   chisquare_effectsize

.. currentmodule:: statsmodels.stats.diagnostic

.. autosummary::
   :toctree: generated/

   normal_ad
   kstest_normal
   lilliefors

非参数检验
--------------------

.. module:: statsmodels.sandbox.stats.runs
   :synopsis: Experimental statistical methods and tests to analyze runs

.. currentmodule:: statsmodels.sandbox.stats.runs

.. autosummary::
   :toctree: generated/

   mcnemar
   symmetry_bowker
   median_test_ksample
   runstest_1samp
   runstest_2samp
   cochrans_q
   Runs

.. module:: statsmodels.stats.descriptivestats
   :synopsis: Descriptive statistics

.. currentmodule:: statsmodels.stats.descriptivestats

.. autosummary::
   :toctree: generated/

   sign_test

.. _interrater:

评估的可靠性和协议
------------------------------------

statsmodels 当前可用于跨界协议度量和检验的主要功能有 Cohen's Kappa。
 Fleiss' Kappa 目前仅作为一种措施执行，而没有相关的结果统计。

.. module:: statsmodels.stats.inter_rater
.. currentmodule:: statsmodels.stats.inter_rater

.. autosummary::
   :toctree: generated/

   cohens_kappa
   fleiss_kappa
   to_table
   aggregate_raters

多次检验和比较的程序
-------------------------------------------------

`multipletests` 是用于 p-value 校正的功能，还包括基于 `fdrcorrection` 中的 fdr 的 p-value 校正
`tukeyhsd` 同时执行检验与比较（独立）均值，这三个功能已得到了验证，GroupsStats 和 MultiComparison 
是对多次比较的便捷的类，类似于一种方差分析，但仍在开发中。

.. module:: statsmodels.sandbox.stats.multicomp
   :synopsis: 用于在进行多次比较时控制尺寸的实验方法


.. currentmodule:: statsmodels.stats.multitest

.. autosummary::
   :toctree: generated/

   multipletests
   fdrcorrection

.. currentmodule:: statsmodels.sandbox.stats.multicomp

.. autosummary::
   :toctree: generated/

   GroupsStats
   MultiComparison
   TukeyHSDResults

.. module:: statsmodels.stats.multicomp
   :synopsis: 执行多个比较时控制大小的方法

.. currentmodule:: statsmodels.stats.multicomp

.. autosummary::
   :toctree: generated/

   pairwise_tukeyhsd

.. module:: statsmodels.stats.multitest
   :synopsis: 多次检验的 p-value 和 FDR 调整

.. currentmodule:: statsmodels.stats.multitest

.. autosummary::
   :toctree: generated/

   local_fdr
   fdrcorrection_twostage
   NullDistribution
   RegressionFDR

.. module:: statsmodels.stats.knockoff_regeffects
   :synopsis: Regression Knock-Off Effects

.. currentmodule:: statsmodels.stats.knockoff_regeffects

.. autosummary::
   :toctree: generated/

   CorrelationEffects
   OLSEffects
   ForwardEffects
   OLSEffects
   RegModelEffects

以下功能尚未公开

.. currentmodule:: statsmodels.sandbox.stats.multicomp

.. autosummary::
   :toctree: generated/

   varcorrection_pairs_unbalanced
   varcorrection_pairs_unequal
   varcorrection_unbalanced
   varcorrection_unequal

   StepDown
   catstack
   ccols
   compare_ordered
   distance_st_range
   ecdf
   get_tukeyQcrit
   homogeneous_subsets
   maxzero
   maxzerodown
   mcfdr
   qcrit
   randmvn
   rankdata
   rejectionline
   set_partition
   set_remove_subs
   tiecorrect

.. _tost:

具有频率权重的基本统计量和t检验
---------------------------------------------------

除了基本统计数据（如具有案例权重的数据的均值，方差，协方差和相关性）外，
此处的类还提供一个和两个均值样本检验。t检验比scipy.stats中的检验有更多选择，
但对阵列形状的限制更多。根据与t检验相同的假设提供均值的置信区间。

此外，均值相等性检验可用于一个样本和两个成对或独立样本。
这些检验基于TOST（两个单面检验），作为零假设，即手段彼此之间并不“接近”。

.. module:: statsmodels.stats.weightstats
   :synopsis: Weighted statistics

.. currentmodule:: statsmodels.stats.weightstats

.. autosummary::
   :toctree: generated/

   DescrStatsW
   CompareMeans
   ttest_ind
   ttost_ind
   ttost_paired
   ztest
   ztost
   zconfint

weightstats 还包含基于汇总数据的测试和置信区间。

.. currentmodule:: statsmodels.stats.weightstats

.. autosummary::
   :toctree: generated/

   _tconfint_generic
   _tstat_generic
   _zconfint_generic
   _zstat_generic
   _zstat_generic2


功效和样本数量计算
----------------------------------

 :mod:`power` 模块目前实现功率和样本大小计算的t检验, 正常的基于检验，F-检验和配合检验的卡方优度，
 该实现是基于类的，但该模块还提供了三种快捷功能， ``tt_solve_power``, ``tt_ind_solve_power`` 和
``zt_ind_solve_power`` 求解功效方程中的任一参数。


.. module:: statsmodels.stats.power
   :synopsis: Power and size calculations for common tests

.. currentmodule:: statsmodels.stats.power

.. autosummary::
   :toctree: generated/

   TTestIndPower
   TTestPower
   GofChisquarePower
   NormalIndPower
   FTestAnovaPower
   FTestPower
   tt_solve_power
   tt_ind_solve_power
   zt_ind_solve_power


.. _proportion_stats:

比例
----------


 NormalIndPower 可以与假设检验，置信区间和比例效应大小共同来使用。

.. module:: statsmodels.stats.proportion
   :synopsis: Tests for proportions

.. currentmodule:: statsmodels.stats.proportion

.. autosummary::
   :toctree: generated

   proportion_confint
   proportion_effectsize

   binom_test
   binom_test_reject_interval
   binom_tost
   binom_tost_reject_interval

   multinomial_proportions_confint

   proportions_ztest
   proportions_ztost
   proportions_chisquare
   proportions_chisquare_allpairs
   proportions_chisquare_pairscontrol

   proportion_effectsize
   power_binom_tost
   power_ztost_prop
   samplesize_confint_proportion


Moment 助手
--------------

当缺少值时，相关性或协方差矩阵可能不是正半定的。
可以使用以下三个函数查找正定且接近原始矩阵的相关或协方差矩阵。

.. module:: statsmodels.stats.correlation_tools
   :synopsis: 确保相关性为正半定性的程序

.. currentmodule:: statsmodels.stats.correlation_tools

.. autosummary::
   :toctree: generated/

   corr_clipped
   corr_nearest
   corr_nearest_factor
   corr_thresholded
   cov_nearest
   cov_nearest_factor_homog
   FactoredPSDMatrix
   kernel_covariance


这些实用程序功能可在中央和非中央力矩，偏斜，峰度和累积量之间进行转换。

.. module:: statsmodels.stats.moment_helpers
   :synopsis: Tools for converting moments

.. currentmodule:: statsmodels.stats.moment_helpers

.. autosummary::
   :toctree: generated/

   cum2mc
   mc2mnc
   mc2mvsk
   mnc2cum
   mnc2mc
   mnc2mvsk
   mvsk2mc
   mvsk2mnc
   cov2corr
   corr2cov
   se_cov


调解分析
------------------

调解分析关注三个关键变量之间的关系： 'outcome', 'treatment', 和 'mediator'. 
由于调解分析是因果推理的一种形式，因此涉及一些难以验证或无法验证的假设。
理想情况下，调解分析是在这样的实验环境中进行的，在这种实验中，治疗是随机分配的。
人们通常使用观察数据进行调解分析，在这种情况下，治疗可被视为“暴露”。在观察环境中，
调解分析背后的假设更加难以验证。

.. module:: statsmodels.stats.mediation
   :synopsis: 中介分析

.. currentmodule:: statsmodels.stats.mediation

.. autosummary::
   :toctree: generated/

   Mediation
   MediationResults


Oaxaca-Blinder 分解
----------------------------
 
 Oaxaca-Blinder 或某些人称之为 Blinder-Oaxaca 的分解试图解释群体均值的差距。
 它使用两个给定回归方程的线性模型来显示回归系数和已知数据所解释的内容，
 以及使用相同数据无法解释的内容。 Oaxaca-Blinder 分解有两种类型，分别是两倍和三倍分解，
 在经济学文献中都可以并且用于讨论群体差异。此方法有助于对歧视或未观察到的影响进行分类。
 此函数尝试将 STATA 中的 oaxaca 命令功能移植到 Python 。

.. module:: statsmodels.stats.oaxaca
   :synopsis: Oaxaca-Blinder Decomposition

.. currentmodule:: statsmodels.stats.oaxaca

.. autosummary::
   :toctree: generated/

   OaxacaBlinder
   OaxacaResults