# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook generic_mle.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 极大似然估计 (广义模型)

# 本教程说明了如何在 statsmodels 中快速实现新的极大似然模型。 我们举两个例子：
#
# 1. 两分类因变量的 Probit 模型
# 2. 计数数据的负二项式模型
#
# GenericLikelihoodModel 类通过提供诸如自动数值微分和 ``scipy'' 优化功能的统一接口之类的工具简化了流程。 
# 使用“ statsmodels”，用户只需通过 "plugging-in" log-似然函数 就可以拟合新的 MLE 模型。


# ## 例 1: Probit 模型

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

# 与 ``statsmodels`` 一同发布的 ``Spector`` 数据集. 你可以像这样访问因变量 (``endog``) 的值向量和回归矩阵 (``exog``) :

data = sm.datasets.spector.load_pandas()
exog = data.exog
endog = data.endog
print(sm.datasets.spector.NOTE)
print(data.exog.head())

# 它们是我们在回归矩阵中添加的一个常数项:

exog = sm.add_constant(exog, prepend=True)

# 创建自己的似然模型，您只需要通过 loglike 方法即可。


class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q * np.dot(exog, params)).sum()


# 估算模型并输出 summary:

sm_probit_manual = MyProbit(endog, exog).fit()
print(sm_probit_manual.summary())

# 将 Probit 的实现与 ``statsmodels``' 的 "canned" 的实现相比较:

sm_probit_canned = sm.Probit(endog, exog).fit()

print(sm_probit_canned.params)
print(sm_probit_manual.params)

print(sm_probit_canned.cov_params())
print(sm_probit_manual.cov_params())

# 请注意，``GenericMaximumLikelihood`` 类提供了自动微分，因此我们不必提供 Hessian 或 Score 函数即可计算协方差估计值。


#
#
# ## 例 2: 计数数据的负二项式模型
#
# 研究计数数据的负二项式模型，其 log-likelihood 对数似然函数(NB-2 型)可以表示为:
#
# $$
#     \mathcal{L}(\beta_j; y, \alpha) = \sum_{i=1}^n y_i ln
#     \left ( \frac{\alpha exp(X_i'\beta)}{1+\alpha exp(X_i'\beta)} \right
# ) -
#     \frac{1}{\alpha} ln(1+\alpha exp(X_i'\beta)) + ln \Gamma (y_i +
# 1/\alpha) - ln \Gamma (y_i+1) - ln \Gamma (1/\alpha)
# $$
#
# 包含回归矩阵 $X$, 系数 $\beta$ 的向量以及负二项式异质性参数 $\alpha$.
#
# 使用 ``scipy`` 中 的``nbinom`` 分布 , 我们可以将似然简写成:
#

import numpy as np
from scipy.stats import nbinom


def _ll_nb2(y, X, beta, alph):
    mu = np.exp(np.dot(X, beta))
    size = 1 / alph
    prob = size / (size + mu)
    ll = nbinom.logpmf(y, size, prob)
    return ll


# ### 新模型类
#
# 我们创建一个继承 ``GenericLikelihoodModel`` 类的新模型类

from statsmodels.base.model import GenericLikelihoodModel


class NBin(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(NBin, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        alph = params[-1]
        beta = params[:-1]
        ll = _ll_nb2(self.endog, self.exog, beta, alph)
        return -ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # 我们还有一个另增参数，需要将它添加到 summary
        self.exog_names.append('alpha')
        if start_params == None:
            # 合理的起始值
            start_params = np.append(np.zeros(self.exog.shape[1]), .5)
            # 截距
            start_params[-2] = np.log(self.endog.mean())
        return super(NBin, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)


# 请注意一下两件重要的事项:
#
# +``nloglikeobs``：该函数会在数据集中的每个观测值（即endog / X矩阵的行）中返回一个负对数似然函数评估。
# +``start_params``：需要提供一维起始值数组。 此数组的大小取决于将在优化中使用的参数数量。
#
# That's it! You're done!
#
# ### 使用范例
#
# [Medpar](https://raw.githubusercontent.com/vincentarelbundock/Rdatas
# ets/doc/COUNT/medpar.html)
# 数据集是以 CSV 格式托管在 [Rdatasets 存储库](https://ra
# w.githubusercontent.com/vincentarelbundock/Rdatasets). 我们使用 [Pandas 库](https://pandas.pydata.org)中的 
# ``read_csv`` 函数加载到内存中，然后打印前几列数据：
#

import statsmodels.api as sm

medpar = sm.datasets.get_rdataset("medpar", "COUNT", cache=True).data

medpar.head()

# 我们感兴趣的模型具有一个非负整数向量作为因变量（``los''），以及5个回归变量：``Intercept''，``type2''，``type3''，``hmo ``， ``white``。
#
# 为了进行估计，我们需要创建两个变量来保存我们的回归变量和结果变量。 这些可以是 ndarray 或 pandas 对象。


y = medpar.los
X = medpar[["type2", "type3", "hmo", "white"]].copy()
X["constant"] = 1

# 然后，我们拟合模型并提取一些信息：

mod = NBin(y, X)
res = mod.fit()

#  提取参数估算值，标准误差，p-values, AIC 等.:

print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('P-values: ', res.pvalues)
print('AIC: ', res.aic)

# 同样，您可以通过键入 ``dir(res)`` 来获得可用信息的完整列表。
# 我们还可以查看 summary 的估算结果。

print(res.summary())

# ### 测试

# 我们可以使用 statsmodels 中的负二项式模型来检查结果，该模型使用分析得分函数和 Hessian 。


res_nbin = sm.NegativeBinomial(y, X).fit(disp=0)
print(res_nbin.summary())

print(res_nbin.params)

print(res_nbin.bse)

# 或者我们可以将上述结果与使用 R 的 MASS 获得的结果进行比较：
#
#     url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdataset
# s/csv/COUNT/medpar.csv'
#     medpar = read.csv(url)
#     f = los~factor(type)+hmo+white
#
#     library(MASS)
#     mod = glm.nb(f, medpar)
#     coef(summary(mod))
#                      Estimate Std. Error   z value      Pr(>|z|)
#     (Intercept)    2.31027893 0.06744676 34.253370 3.885556e-257
#     factor(type)2  0.22124898 0.05045746  4.384861  1.160597e-05
#     factor(type)3  0.70615882 0.07599849  9.291748  1.517751e-20
#     hmo           -0.06795522 0.05321375 -1.277024  2.015939e-01
#     white         -0.12906544 0.06836272 -1.887951  5.903257e-02
#
# ### 数值精度
#
# ``statsmodels`` 通用 MLE 和 ``R`` 参数估计值一致，直到第四个小数点为止。 但是，标准误差只能精确到小数点后第二位。 
# 这种差异是由于 Hessian 数值估算结果不精确导致的。 在这种情况下，``MASS`` 和``statsmodels`` 标准误差估算值之间的
# 差异本质上是不相关的，但它突出了一个事实：即想要非常精确的估算值的用户在使用数值时可能不能总是依赖使用默认设置。 
# 在这种情况下，最好将分析派生类与 ``LikelihoodModel`` 类一起使用。

