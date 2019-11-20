# DO NOT EDIT
# Autogenerated from the notebook robust_models_1.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

#!/usr/bin/env python
# coding: utf-8

# # 用于稳健线性建模的 M-估计器

from statsmodels.compat import lmap
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm

# * M估计器使函数最小化
#
# $$Q(e_i, \rho) = \sum_i~\rho \left (\frac{e_i}{s}\right )$$
#
# 其中 $\rho$ 是一个残差的对称函数
#
# *  $\rho$ 的作用是减少异常值的影响，
# * $s$ 是一个规模估计
# * 稳健估计 $\hat{\beta}$ 是通过迭代重新加权最小二乘法来计算

# * 我们有几种可用的加权函数可供选择

norms = sm.robust.norms


def plot_weights(support, weights_func, xlabels, xticks):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(support, weights_func(support))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_ylim(-.1, 1.1)
    return ax


# ### Andrew's Wave

help(norms.AndrewWave.weights)

a = 1.339
support = np.linspace(-np.pi * a, np.pi * a, 100)
andrew = norms.AndrewWave(a=a)
plot_weights(support, andrew.weights, ['$-\pi*a$', '0', '$\pi*a$'],
             [-np.pi * a, 0, np.pi * a])

# ### 安德鲁波 Hampel's 17A

help(norms.Hampel.weights)

c = 8
support = np.linspace(-3 * c, 3 * c, 1000)
hampel = norms.Hampel(a=2., b=4., c=c)
plot_weights(support, hampel.weights, ['3*c', '0', '3*c'], [-3 * c, 0, 3 * c])

# ### Huber 的 t 范数

help(norms.HuberT.weights)

t = 1.345
support = np.linspace(-3 * t, 3 * t, 1000)
huber = norms.HuberT(t=t)
plot_weights(support, huber.weights, ['-3*t', '0', '3*t'], [-3 * t, 0, 3 * t])

# ### 最小二乘法

help(norms.LeastSquares.weights)

support = np.linspace(-3, 3, 1000)
lst_sq = norms.LeastSquares()
plot_weights(support, lst_sq.weights, ['-3', '0', '3'], [-3, 0, 3])

# ### Ramsay's Ea

help(norms.RamsayE.weights)

a = .3
support = np.linspace(-3 * a, 3 * a, 1000)
ramsay = norms.RamsayE(a=a)
plot_weights(support, ramsay.weights, ['-3*a', '0', '3*a'], [-3 * a, 0, 3 * a])

# ### Trimmed Mean

help(norms.TrimmedMean.weights)

c = 2
support = np.linspace(-3 * c, 3 * c, 1000)
trimmed = norms.TrimmedMean(c=c)
plot_weights(support, trimmed.weights, ['-3*c', '0', '3*c'],
             [-3 * c, 0, 3 * c])

# ### Tukey's Biweight

help(norms.TukeyBiweight.weights)

c = 4.685
support = np.linspace(-3 * c, 3 * c, 1000)
tukey = norms.TukeyBiweight(c=c)
plot_weights(support, tukey.weights, ['-3*c', '0', '3*c'], [-3 * c, 0, 3 * c])

# ### 标度估计器

# * 位置的稳健估计

x = np.array([1, 2, 3, 4, 500])

# * 均值不是位置的稳健估计

x.mean()

# * 另一方面，中位数是一个可靠的估算器，其分解点为50％

np.median(x)

# * Analogously for the scale 类似于标准化
# * 标准偏差不稳健

x.std()

# 中位数的绝对偏差
#
# $$ median_i |X_i - median_j(X_j)|) $$

# 中位数标准化的绝对偏差是 $\hat{\sigma}$
#
# $$\hat{\sigma}=K \cdot MAD$$
#
# 其中 $K$ 取决于分布.例如，对于正态分布，
#
# $$K = \Phi^{-1}(.75)$$

stats.norm.ppf(.75)

print(x)

sm.robust.scale.mad(x)

np.array([1, 2, 3, 4, 5.]).std()

# * 鲁棒线性模型的默认值为MAD
# * 另一个受欢迎的选择是 Huber's proposal 2

np.random.seed(12345)
fat_tails = stats.t(6).rvs(40)

kde = sm.nonparametric.KDEUnivariate(fat_tails)
kde.fit()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(kde.support, kde.density)

print(fat_tails.mean(), fat_tails.std())

print(stats.norm.fit(fat_tails))

print(stats.t.fit(fat_tails, f0=6))

huber = sm.robust.scale.Huber()
loc, scale = huber(fat_tails)
print(loc, scale)

sm.robust.mad(fat_tails)

sm.robust.mad(fat_tails, c=stats.t(6).ppf(.75))

sm.robust.scale.mad(fat_tails)

# ### Duncan 的职业威望数据——  异常值的M-估计器

from statsmodels.graphics.api import abline_plot
from statsmodels.formula.api import ols, rlm

prestige = sm.datasets.get_rdataset("Duncan", "carData", cache=True).data

print(prestige.head(10))

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(211, xlabel='Income', ylabel='Prestige')
ax1.scatter(prestige.income, prestige.prestige)
xy_outlier = prestige.loc['minister', ['income', 'prestige']]
ax1.annotate('Minister', xy_outlier, xy_outlier + 1, fontsize=16)
ax2 = fig.add_subplot(212, xlabel='Education', ylabel='Prestige')
ax2.scatter(prestige.education, prestige.prestige)

ols_model = ols('prestige ~ income + education', prestige).fit()
print(ols_model.summary())

infl = ols_model.get_influence()
student = infl.summary_frame()['student_resid']
print(student)

print(student.loc[np.abs(student) > 2])

print(infl.summary_frame().loc['minister'])

sidak = ols_model.outlier_test('sidak')
sidak.sort_values('unadj_p', inplace=True)
print(sidak)

fdr = ols_model.outlier_test('fdr_bh')
fdr.sort_values('unadj_p', inplace=True)
print(fdr)

rlm_model = rlm('prestige ~ income + education', prestige).fit()
print(rlm_model.summary())

print(rlm_model.weights)

# ### Hertzprung Russell data for Star Cluster CYG 0B1 - Leverage Points

# * 数据是关于天鹅座方向的 47 颗恒星的光度和温度。

dta = sm.datasets.get_rdataset("starsCYG", "robustbase", cache=True).data

from matplotlib.patches import Ellipse
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(
    111,
    xlabel='log(Temp)',
    ylabel='log(Light)',
    title='Hertzsprung-Russell Diagram of Star Cluster CYG OB1')
ax.scatter(*dta.values.T)
# 突出异常值
e = Ellipse((3.5, 6), .2, 1, alpha=.25, color='r')
ax.add_patch(e)
ax.annotate(
    'Red giants',
    xy=(3.6, 6),
    xytext=(3.8, 6),
    arrowprops=dict(facecolor='black', shrink=0.05, width=2),
    horizontalalignment='left',
    verticalalignment='bottom',
    clip_on=True,  # 缩减轴边界框
    fontsize=16,
)
# 用它们的索引注释这些
for i, row in dta.loc[dta['log.Te'] < 3.8].iterrows():
    ax.annotate(i, row, row + .01, fontsize=14)
xlim, ylim = ax.get_xlim(), ax.get_ylim()

from IPython.display import Image
Image(filename='star_diagram.png')

y = dta['log.light']
X = sm.add_constant(dta['log.Te'], prepend=True)
ols_model = sm.OLS(y, X).fit()
abline_plot(model_results=ols_model, ax=ax)

rlm_mod = sm.RLM(y, X, sm.robust.norms.TrimmedMean(.5)).fit()
abline_plot(model_results=rlm_mod, ax=ax, color='red')

# *为什么？ 因为 M-估计量处理杠杆点并不稳健。

infl = ols_model.get_influence()

h_bar = 2 * (ols_model.df_model + 1) / ols_model.nobs
hat_diag = infl.summary_frame()['hat_diag']
hat_diag.loc[hat_diag > h_bar]

sidak2 = ols_model.outlier_test('sidak')
sidak2.sort_values('unadj_p', inplace=True)
print(sidak2)

fdr2 = ols_model.outlier_test('fdr_bh')
fdr2.sort_values('unadj_p', inplace=True)
print(fdr2)

# * 让我们删除那一行

l = ax.lines[-1]
l.remove()
del l

weights = np.ones(len(X))
weights[X[X['log.Te'] < 3.8].index.values - 1] = 0
wls_model = sm.WLS(y, X, weights=weights).fit()
abline_plot(model_results=wls_model, ax=ax, color='green')

# * MM-估算器可解决此类问题，但很遗憾的是目前我们还没有这些估算器。
# * 它正在开发当中，但它给出了一个好借口是笔记文档在 R 单元格魔法

yy = y.values[:, None]
xx = X['log.Te'].values[:, None]

print(params)

abline_plot(intercept=params[0], slope=params[1], ax=ax, color='red')

# ### 练习: M-估计器的分解点

np.random.seed(12345)
nobs = 200
beta_true = np.array([3, 1, 2.5, 3, -4])
X = np.random.uniform(-20, 20, size=(nobs, len(beta_true) - 1))
# 在前面添加一个常量
X = sm.add_constant(X, prepend=True)  # np.c_[np.ones(nobs), X]
mc_iter = 500
contaminate = .25  # 污染的响应变量百分比

all_betas = []
for i in range(mc_iter):
    y = np.dot(X, beta_true) + np.random.normal(size=200)
    random_idx = np.random.randint(0, nobs, size=int(contaminate * nobs))
    y[random_idx] = np.random.uniform(-750, 750)
    beta_hat = sm.RLM(y, X).fit().params
    all_betas.append(beta_hat)

all_betas = np.asarray(all_betas)
se_loss = lambda x: np.linalg.norm(x, ord=2)**2
se_beta = lmap(se_loss, all_betas - beta_true)

# #### 平方误差损失

np.array(se_beta).mean()

all_betas.mean(0)

beta_true

se_loss(all_betas.mean(0) - beta_true)
