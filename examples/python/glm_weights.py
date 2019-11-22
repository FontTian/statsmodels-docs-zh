# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook glm_weights.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 加权广义线性模型

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ## 加权GLM：泊松响应数据
#
# ### 加载数据
#
# 在此示例中，我们将使用少数几个外生变量的婚外恋数据集来预测婚外恋比率。
# 
#
# 权重将被生成以表明 `freq_weights` 等同于重复记录数据。 另一方面， `var_weights` 等效于汇总数据。

print(sm.datasets.fair.NOTE)

# 将数据加载到 pandas 数据框中。

data = sm.datasets.fair.load_pandas().data

#  因变量（内生变量）是``外遇''

data.describe()

data[:3]

# 在下文中，我们将主要与Poisson合作。 当使用十进制事务时，我们把它们转换为整数使他们具有计数分布。
# 

data["affairs"] = np.ceil(data["affairs"])
data[:3]

(data["affairs"] == 0).mean()

np.bincount(data["affairs"].astype(int))

# ## 浓缩和汇总观测
#
# 我们的原始数据集中有6366个观测值。 当我们只考虑一些选定的变量时，则只有较少的观察值。 
# 在下文中，我们以两种方式组合观察值，首先，将所有变量值相同的观察值相结合，
# 其次，我们将解释性变量相同的观察值相结合。
# 

# ### 具有唯一观测值的数据集
#
# 我们使用 pandas 的 groupby 来组合相同的观测值，并创建一个新的变量 `freq` ，该变量计算对应行中有多少观测值。

data2 = data.copy()
data2['const'] = 1
dc = data2['affairs rate_marriage age yrs_married const'.split()].groupby(
    'affairs rate_marriage age yrs_married'.split()).count()
dc.reset_index(inplace=True)
dc.rename(columns={'const': 'freq'}, inplace=True)
print(dc.shape)
dc.head()

# ### 具有唯一解释变量的数据集（exog）
#
# 对于下一个数据集，我们将合并具有相同解释变量值的观察值。 但是，由于响应变量在组合观测之间可能有所不同，
# 因此我们计算所有组合观测响应变量的平均值和总和。

# 我们再次使用 pandas 的 ``groupby`` 来组合观察值并创建新变量。 我们还将 ``MultiIndex`` 展平为一个简单的索引。


gr = data['affairs rate_marriage age yrs_married'.split()].groupby(
    'rate_marriage age yrs_married'.split())
df_a = gr.agg(['mean', 'sum', 'count'])


def merge_tuple(tpl):
    if isinstance(tpl, tuple) and len(tpl) > 1:
        return "_".join(map(str, tpl))
    else:
        return tpl


df_a.columns = df_a.columns.map(merge_tuple)
df_a.reset_index(inplace=True)
print(df_a.shape)
df_a.head()

# 组合观察值之后，将有467个唯一观察值的数据框 `dc` 和具有130个观察值的解释性变量的唯一值的数据框 `df_a`
#

print('number of rows: \noriginal, with unique observations, with unique exog')
data.shape[0], dc.shape[0], df_a.shape[0]

# ## 分析
#
# 在下文中，我们将原始数据 GLM-泊松 的结果与组合观测值的模型进行比较，在组合观测值中，多重性或聚集性是由权重或暴露给出的。
#
#
# ### 原始数据

glm = smf.glm(
    'affairs ~ rate_marriage + age + yrs_married',
    data=data,
    family=sm.families.Poisson())
res_o = glm.fit()
print(res_o.summary())

res_o.pearson_chi2 / res_o.df_resid

# ### 压缩数据（具有频率的唯一观测值）
#
# 组合相同的观察值并使用频率权重考虑多个观察值，即可得出完全相同的结果。 
# 当我们想要获得有关的观测信息而不是所有相同观测的汇总时，某些结果属性将有所不同。
# 例如，不考虑残差 ``freq_weights`` 的情况。


glm = smf.glm(
    'affairs ~ rate_marriage + age + yrs_married',
    data=dc,
    family=sm.families.Poisson(),
    freq_weights=np.asarray(dc['freq']))
res_f = glm.fit()
print(res_f.summary())

res_f.pearson_chi2 / res_f.df_resid

# ### 使用 ``var_weights`` 而不是 ``freq_weights`` 来压缩
#
# 下一步，我们将 ``var_weights`` 与 ``freq_weights`` 进行比较。 当内生变量反映平均值而不是相同的观察值时，通常是包括“ var_weights”。
# 我看不出产生相同结果的理论原因（大体上）。

# 这会产生相同的结果，但 ``df_resid``  与 ``freq_weights`` 示例有所不同，因为 ``var_weights`` 无法改变有效观测值的数量。
#

glm = smf.glm(
    'affairs ~ rate_marriage + age + yrs_married',
    data=dc,
    family=sm.families.Poisson(),
    var_weights=np.asarray(dc['freq']))
res_fv = glm.fit()
print(res_fv.summary())

# 由于错误的``df_resid''，从结果计算出的 dispersion 不正确。
# 如果我们使用原始的df_resid是正确的。

res_fv.pearson_chi2 / res_fv.df_resid, res_f.pearson_chi2 / res_f.df_resid

# ### 聚合或平均数据（解释变量的唯一值）
#
# 对于这些情况，我们合并了具有相同解释变量值的观察值。 相应的响应变量是总和或平均值。
#
# #### 使用 ``exposure``
#
# 如果我们的因变量是所有组合观测值的响应之和，则在泊松假设下，分布保持不变，
# 但是我们通过聚合观测值表示的个体数量给定可变的 `exposure`。
#
# 参数估计值和参数的协方差与原始数据相同，但对数似然，偏差和 Pearson 卡方不同
# 

glm = smf.glm(
    'affairs_sum ~ rate_marriage + age + yrs_married',
    data=df_a,
    family=sm.families.Poisson(),
    exposure=np.asarray(df_a['affairs_count']))
res_e = glm.fit()
print(res_e.summary())

res_e.pearson_chi2 / res_e.df_resid

# #### 使用 var_weights
#
# 我们还可以使用因变量的所有组合值的平均值。 在这种情况下，方差与一个组合观察所反映的总暴露量的倒数有关。

glm = smf.glm(
    'affairs_mean ~ rate_marriage + age + yrs_married',
    data=df_a,
    family=sm.families.Poisson(),
    var_weights=np.asarray(df_a['affairs_count']))
res_a = glm.fit()
print(res_a.summary())

# ### 比较
#
# 我们在上面的摘要打印中看到，带有相关Wald推断的 ``params'' 和 ``cov_params'' 在各个版本之间是一致的。
# 在下面比较各个版本的各个结果属性时，我们对此进行了总结。
#
# 参数估计 `params`，参数 `bse` 的标准误差和参数 `pvalues`（对于参数为零的检验）全部一致。 
# 但是，似然和拟合优度统计分析。 `llf`，`deviance` 和 `pearson_chi2` 仅部分一致。 
# 具体而言，汇总版本与使用原始数据的结果不一致。
# 
# **警告**: 在以后的版本中， `llf`, `deviance` 和 `pearson_chi2` 可能仍会。改变
#
# 对于解释变量唯一值之和与平均值响应变量都有正确的似然解释。 但是，此解释未反映在这三个统计数据中。
# 从计算上讲，这可能是由于使用聚合数据时没有调整。 
# 但是，从理论上讲，我们可以考虑在这些情况下，特别是对于错误指定情况下的 `var_weights` ，（当似然分析不合适时，应将结果解释为准似然估计）。
#  ``var_weights'' 的定义不明确，因为它们可以用于具有正确指定的似然性的平均值以及在准可能性情况下的方差调整。 我们目前不尝试匹配似然性规范。 
# 但是，在下一节中，我们表明当假设正确指定了基础模型时，似然比类型检验对于所有聚合版本仍会产生相同的结果。


results_all = [res_o, res_f, res_e, res_a]
names = 'res_o res_f res_e res_a'.split()

pd.concat([r.params for r in results_all], axis=1, keys=names)

pd.concat([r.bse for r in results_all], axis=1, keys=names)

pd.concat([r.pvalues for r in results_all], axis=1, keys=names)

pd.DataFrame(
    np.column_stack(
        [[r.llf, r.deviance, r.pearson_chi2] for r in results_all]),
    columns=names,
    index=['llf', 'deviance', 'pearson chi2'])

# ### 似然比类型检验
#
# 我们从上看到，聚合数据和原始的个体数据之间的似然和相关统计数据不一致。 下面我们将说明
# 似然比检验和偏差差异在各个版本中是一致的，但是Pearson卡方却不同的情况。
#
# 和以前一样：这还不够清楚，可能会改变。
#
# 作为测试用例，我们删除 `age` 变量，并计算似然比类型统计量作为缩小或约束模型与完全或非约束模型之间的差异。


# #### 原始观测值和频率权重

glm = smf.glm(
    'affairs ~ rate_marriage + yrs_married',
    data=data,
    family=sm.families.Poisson())
res_o2 = glm.fit()
#print(res_f2.summary())
res_o2.pearson_chi2 - res_o.pearson_chi2, res_o2.deviance - res_o.deviance, res_o2.llf - res_o.llf

glm = smf.glm(
    'affairs ~ rate_marriage + yrs_married',
    data=dc,
    family=sm.families.Poisson(),
    freq_weights=np.asarray(dc['freq']))
res_f2 = glm.fit()
#print(res_f2.summary())
res_f2.pearson_chi2 - res_f.pearson_chi2, res_f2.deviance - res_f.deviance, res_f2.llf - res_f.llf

# #### 聚合数据: ``exposure`` 和 ``var_weights``
#
# 警告: LR 检验与原始观测值一致的情形， ``pearson_chi2`` 却有所不同且有错误标识。


glm = smf.glm(
    'affairs_sum ~ rate_marriage + yrs_married',
    data=df_a,
    family=sm.families.Poisson(),
    exposure=np.asarray(df_a['affairs_count']))
res_e2 = glm.fit()
res_e2.pearson_chi2 - res_e.pearson_chi2, res_e2.deviance - res_e.deviance, res_e2.llf - res_e.llf

glm = smf.glm(
    'affairs_mean ~ rate_marriage + yrs_married',
    data=df_a,
    family=sm.families.Poisson(),
    var_weights=np.asarray(df_a['affairs_count']))
res_a2 = glm.fit()
res_a2.pearson_chi2 - res_a.pearson_chi2, res_a2.deviance - res_a.deviance, res_a2.llf - res_a.llf

# ### 探讨 Pearson卡方统计
#
# 首先，我们进行一些合理性检验，以确保在计算 `pearson_chi2` 和 `resid_pearson` 时没有基本的错误。

res_e2.pearson_chi2, res_e.pearson_chi2, (res_e2.resid_pearson
                                          **2).sum(), (res_e.resid_pearson
                                                       **2).sum()

res_e._results.resid_response.mean(), res_e.model.family.variance(
    res_e.mu)[:5], res_e.mu[:5]

(res_e._results.resid_response**2 / res_e.model.family.variance(
    res_e.mu)).sum()

res_e2._results.resid_response.mean(), res_e2.model.family.variance(
    res_e2.mu)[:5], res_e2.mu[:5]

(res_e2._results.resid_response**2 / res_e2.model.family.variance(
    res_e2.mu)).sum()

(res_e2._results.resid_response**2).sum(), (res_e._results.resid_response
                                            **2).sum()

# 错误标识的一种可能原因是我们要减去的二次项被不同分母作除法。在某些相关情况下，文献中的建议是使用公分母。
# 我们可以在完全模型和简化模型中使用相同方差假设来比较皮尔逊卡方统计量。
# 
#
# 在这种情况下，我们在所有版本的简化模型和完整模型之间都获得了相同的皮尔逊卡方标度差异。 (问题 [#3616](https://github.com/statsmodels/statsmodels/issues/3616) is
# 将做进一步的追踪。)

((res_e2._results.resid_response**2 - res_e._results.resid_response**2) /
 res_e2.model.family.variance(res_e2.mu)).sum()

((res_a2._results.resid_response**2 - res_a._results.resid_response**2) /
 res_a2.model.family.variance(res_a2.mu) * res_a2.model.var_weights).sum()

((res_f2._results.resid_response**2 - res_f._results.resid_response**2) /
 res_f2.model.family.variance(res_f2.mu) * res_f2.model.freq_weights).sum()

((res_o2._results.resid_response**2 - res_o._results.resid_response**2) /
 res_o2.model.family.variance(res_o2.mu)).sum()

# ## 其余内容
#
# 笔记的其余部分包含一些其他检查，可以忽略。

np.exp(res_e2.model.exposure)[:5], np.asarray(df_a['affairs_count'])[:5]

res_e2.resid_pearson.sum() - res_e.resid_pearson.sum()

res_e2.mu[:5]

res_a2.pearson_chi2, res_a.pearson_chi2, res_a2.resid_pearson.sum(
), res_a.resid_pearson.sum()

((res_a2._results.resid_response**2) / res_a2.model.family.variance(res_a2.mu)
 * res_a2.model.var_weights).sum()

((res_a._results.resid_response**2) / res_a.model.family.variance(res_a.mu) *
 res_a.model.var_weights).sum()

((res_a._results.resid_response**2) / res_a.model.family.variance(res_a2.mu) *
 res_a.model.var_weights).sum()

res_e.model.endog[:5], res_e2.model.endog[:5]

res_a.model.endog[:5], res_a2.model.endog[:5]

res_a2.model.endog[:5] * np.exp(res_e2.model.exposure)[:5]

res_a2.model.endog[:5] * res_a2.model.var_weights[:5]

from scipy import stats
stats.chi2.sf(27.19530754604785, 1), stats.chi2.sf(29.083798806764687, 1)

res_o.pvalues

print(res_e2.summary())
print(res_e.summary())

print(res_f2.summary())
print(res_f.summary())
