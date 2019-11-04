"""
加权最小二乘法

扩展示例意在说明 WLS 模型中 r_squared 的意义，对于异常值的处理，与 RLM 和较短的引导程序相比较

"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = sm.datasets.ccard.load()
data.exog = sm.add_constant(data.exog, prepend=False)
ols_fit = sm.OLS(data.endog, data.exog).fit()

# 也许这种拟合的残差取决于收入的平方
incomesq = data.exog[:, 2]
plt.scatter(incomesq, ols_fit.resid)
# @savefig wls_resid_check.png
plt.grid()


# 如果我们认为方差与收入的平方成正比，我们将要按收入对回归进行加权，WLS中的权重参数按其平方根对回归进行加权，
# 并且由于收入进入方程，如果我们有收入，它将变成常数，所以希望运行一种没有明确常数项的回归

wls_fit = sm.WLS(data.endog, data.exog[:, :-1], weights=1/incomesq).fit()

# 但是，这导致估计后的统计解释存在困难。 statsmodels 尚不能很好地处理此问题，但是以下方法可能更合适

# 解释平方和
ess = wls_fit.uncentered_tss - wls_fit.ssr
# r_squared
rsquared = ess/wls_fit.uncentered_tss
# 模型的均方误差
mse_model = ess/(wls_fit.df_model + 1)  # add back the dof of the constant
# f 统计
fvalue = mse_model/wls_fit.mse_resid
# 调整后的 r-squared
rsquared_adj = 1 - (wls_fit.nobs)/(wls_fit.df_resid)*(1-rsquared)


# 试图弄清本示例的意图
# ----------------------------------------------------

# JP: 我需要在看看，即使从回归估计器中剔除体重变量并保留常数项，返回的R_squared依旧很小。下面我们比较了使用平方或权重变量的平方。

# TODO: 需要添加 45度 线到图中
wls_fit3 = sm.WLS(data.endog, data.exog[:, (0, 1, 3, 4)],
                  weights=1/incomesq).fit()
print(wls_fit3.summary())
print('corrected rsquared')
print((wls_fit3.uncentered_tss - wls_fit3.ssr)/wls_fit3.uncentered_tss)
plt.figure()
plt.title('WLS dropping heteroscedasticity variable from regressors')
plt.plot(data.endog, wls_fit3.fittedvalues, 'o')
plt.xlim([0, 2000])
# @savefig wls_drop_het.png
plt.ylim([0, 2000])
print('raw correlation of endog and fittedvalues')
print(np.corrcoef(data.endog, wls_fit.fittedvalues))
print('raw correlation coefficient of endog and fittedvalues squared')
print(np.corrcoef(data.endog, wls_fit.fittedvalues)[0, 1]**2)

# 与鲁棒性回归比较，异方差校正可降低异常值的影响

rlm_fit = sm.RLM(data.endog, data.exog).fit()
plt.figure()
plt.title('using robust for comparison')
plt.plot(data.endog, rlm_fit.fittedvalues, 'o')
plt.xlim([0, 2000])
# @savefig wls_robust_compare.png
plt.ylim([0, 2000])


# 继续将发生什么? 更加系统的查看数据
# ----------------------------------------------------

# 两个辅助函数

def getrsq(fitresult):
    '''计算平方残差，平方和的解释量与总和。

    参数
    ----------
    fitresult : 回归结果类的实例，回归残差与内生变量的数组(resid,endog)

    返回
    -------
    r_squared
    （居中）平方总和(centered) total sum of squares
    解释平方和 explained sum of squares (for centered)
    '''
    if hasattr(fitresult, 'resid') and hasattr(fitresult, 'model'):
        resid = fitresult.resid
        endog = fitresult.model.endog
        nobs = fitresult.nobs
    else:
        resid = fitresult[0]
        endog = fitresult[1]
        nobs = resid.shape[0]

    rss = np.dot(resid, resid)
    tss = np.var(endog)*nobs
    return 1-rss/tss, rss, tss, tss-rss


def index_trim_outlier(resid, k):
    '''返回删除了k个异常值的残差数组的索引

    参数
    ----------
    resid : array_like, 1d
        数据的向量，通常表示回归的残差
    k : int
        移除的异常值数量k

    返回
    -------
    trimmed_index : array, 1d
        删除 k 个异常值的索引数组
    outlier_index : array, 1d
        k 个异常值的索引数组

    注意
    -----

    离群值可以定义为最大的k个观测值的绝对值。
    

    '''
    sort_index = np.argsort(np.abs(resid))
    # 正常值的索引
    trimmed_index = np.sort(sort_index[:-k])
    outlier_index = np.sort(sort_index[-k:])
    return trimmed_index, outlier_index


# 比较 ols、rlm 和 wls 模型是否存在异常值的估计结果差异
# ---------------------------------------------------------------------------

olskeep, olsoutl = index_trim_outlier(ols_fit.resid, 2)
print('ols outliers', olsoutl, ols_fit.resid[olsoutl])
ols_fit_rm2 = sm.OLS(data.endog[olskeep], data.exog[olskeep, :]).fit()
rlm_fit_rm2 = sm.RLM(data.endog[olskeep], data.exog[olskeep, :]).fit()

results = [ols_fit, ols_fit_rm2, rlm_fit, rlm_fit_rm2]
# 注意: 收入残差的平方
for weights in [1/incomesq, 1/incomesq**2, np.sqrt(incomesq)]:
    print('\nComparison OLS and WLS with and without outliers')
    wls_fit0 = sm.WLS(data.endog, data.exog, weights=weights).fit()
    wls_fit_rm2 = sm.WLS(data.endog[olskeep], data.exog[olskeep, :],
                         weights=weights[olskeep]).fit()
    wlskeep, wlsoutl = index_trim_outlier(ols_fit.resid, 2)
    print('2 outliers candidates and residuals')
    print(wlsoutl, wls_fit.resid[olsoutl])
    # 因为 ols 和 wls 模型的离群值相同，所以是多余的：
    #  wls_fit_rm2_ = sm.WLS(data.endog[wlskeep], data.exog[wlskeep, :],
    #                        weights=1/incomesq[wlskeep]).fit()

    print('outliers ols, wls:', olsoutl, wlsoutl)

    print('rsquared')
    print('ols vs ols rm2', ols_fit.rsquared, ols_fit_rm2.rsquared)
    print('wls vs wls rm2', wls_fit0.rsquared, wls_fit_rm2.rsquared)
    print('compare R2_resid  versus  R2_wresid')
    print('ols minus 2', getrsq(ols_fit_rm2)[0],)
    print(getrsq((ols_fit_rm2.wresid, ols_fit_rm2.model.wendog))[0])
    print('wls        ', getrsq(wls_fit)[0],)
    print(getrsq((wls_fit.wresid, wls_fit.model.wendog))[0])

    print('wls minus 2', getrsq(wls_fit_rm2)[0])
    # 下面的 wls_fit_rm2.rsquared 的交叉验证是一致的。
    print(getrsq((wls_fit_rm2.wresid, wls_fit_rm2.model.wendog))[0])
    results.extend([wls_fit0, wls_fit_rm2])

print('     ols             ols_rm2       rlm           rlm_rm2     '
      'wls (lin)    wls_rm2 (lin)   '
      'wls (squ)   wls_rm2 (squ)  '
      'wls (sqrt)   wls_rm2 (sqrt)')
print('Parameter estimates')
print(np.column_stack([r.params for r in results]))
print('R2 original data, next line R2 weighted data')
print(np.column_stack([getattr(r, 'rsquared', None) for r in results]))

print('Standard errors')
print(np.column_stack([getattr(r, 'bse', None) for r in results]))
print('Heteroscedasticity robust standard errors (with ols)')
print('with outliers')
print(np.column_stack([getattr(ols_fit, se, None)
                       for se in ['HC0_se', 'HC1_se', 'HC2_se', 'HC3_se']]))

'''
# ..
# ..
# ..     ols             ols_rm2       rlm           rlm_rm2     wls (lin)    wls_rm2 (lin)   wls (squ)   wls_rm2 (squ)  wls (sqrt)   wls_rm2 (sqrt)
# ..Parameter estimates
# ..[[  -3.08181404   -5.06103843   -4.98510966   -5.34410309   -2.69418516    -3.1305703    -1.43815462   -1.58893054   -3.57074829   -6.80053364]
# .. [ 234.34702702  115.08753715  129.85391456  109.01433492  158.42697752   128.38182357   60.95113284  100.25000841  254.82166855  103.75834726]
# .. [ -14.99684418   -5.77558429   -6.46204829   -4.77409191   -7.24928987    -7.41228893    6.84943071   -3.34972494  -16.40524256   -4.5924465 ]
# .. [  27.94090839   85.46566835   89.91389709   95.85086459   60.44877369    79.7759146    55.9884469    60.97199734   -3.8085159    84.69170048]
# .. [-237.1465136    39.51639838  -15.50014814   31.39771833 -114.10886935   -40.04207242   -6.41976501  -38.83583228 -260.72084271  117.20540179]]
# ..
# ..R2 original data, next line R2 weighted data
# ..[[   0.24357792    0.31745994    0.19220308    0.30527648    0.22861236     0.3112333     0.06573949    0.29366904    0.24114325    0.31218669]]
# ..[[   0.24357791    0.31745994    None          None          0.05936888     0.0679071     0.06661848    0.12769654    0.35326686    0.54681225]]
# ..
# ..-> R2 with weighted data is jumping all over
# ..
# ..standard errors
# ..[[   5.51471653    3.31028758    2.61580069    2.39537089    3.80730631     2.90027255    2.71141739    2.46959477    6.37593755    3.39477842]
# .. [  80.36595035   49.35949263   38.12005692   35.71722666   76.39115431    58.35231328   87.18452039   80.30086861   86.99568216   47.58202096]
# .. [   7.46933695    4.55366113    3.54293763    3.29509357    9.72433732     7.41259156   15.15205888   14.10674821    7.18302629    3.91640711]
# .. [  82.92232357   50.54681754   39.33262384   36.57639175   58.55088753    44.82218676   43.11017757   39.31097542   96.4077482    52.57314209]
# .. [ 199.35166485  122.1287718    94.55866295   88.3741058   139.68749646   106.89445525  115.79258539  105.99258363  239.38105863  130.32619908]]
# ..
# ..robust standard errors (with ols)
# ..with outliers
# ..      HC0_se         HC1_se       HC2_se        HC3_se'
# ..[[   3.30166123    3.42264107    3.4477148     3.60462409]
# .. [  88.86635165   92.12260235   92.08368378   95.48159869]
# .. [   6.94456348    7.19902694    7.19953754    7.47634779]
# .. [  92.18777672   95.56573144   95.67211143   99.31427277]
# .. [ 212.9905298   220.79495237  221.08892661  229.57434782]]
# ..
# ..removing 2 outliers
# ..[[   2.57840843    2.67574088    2.68958007    2.80968452]
# .. [  36.21720995   37.58437497   37.69555106   39.51362437]
# .. [   3.1156149     3.23322638    3.27353882    3.49104794]
# .. [  50.09789409   51.98904166   51.89530067   53.79478834]
# .. [  94.27094886   97.82958699   98.25588281  102.60375381]]
# ..
# ..
# ..
'''  # noqa:E501

# a quick bootstrap analysis
# --------------------------
#
# (我没有从统计上检查这是否完全正确)

# **全样本的 OLS**

nobs, nvar = data.exog.shape
niter = 2000
bootres = np.zeros((niter, nvar*2))

for it in range(niter):
    rind = np.random.randint(nobs, size=nobs)
    endog = data.endog[rind]
    exog = data.exog[rind, :]
    res = sm.OLS(endog, exog).fit()
    bootres[it, :nvar] = res.params
    bootres[it, nvar:] = res.bse

np.set_printoptions(linewidth=200)
print('Bootstrap Results of parameters and parameter standard deviation  OLS')
print('Parameter estimates')
print('median', np.median(bootres[:, :5], 0))
print('mean  ', np.mean(bootres[:, :5], 0))
print('std   ', np.std(bootres[:, :5], 0))

print('Standard deviation of parameter estimates')
print('median', np.median(bootres[:, 5:], 0))
print('mean  ', np.mean(bootres[:, 5:], 0))
print('std   ', np.std(bootres[:, 5:], 0))

plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist(bootres[:, i], 50)
    plt.title('var%d' % i)
# @savefig wls_bootstrap.png
plt.figtext(0.5, 0.935,  'OLS Bootstrap',
            ha='center', color='black', weight='bold', size='large')

# **剔除了异常值的样本 WLS**

data_endog = data.endog[olskeep]
data_exog = data.exog[olskeep, :]
incomesq_rm2 = incomesq[olskeep]

nobs, nvar = data_exog.shape
niter = 500  # a bit slow
bootreswls = np.zeros((niter, nvar*2))

for it in range(niter):
    rind = np.random.randint(nobs, size=nobs)
    endog = data_endog[rind]
    exog = data_exog[rind, :]
    res = sm.WLS(endog, exog, weights=1/incomesq[rind, :]).fit()
    bootreswls[it, :nvar] = res.params
    bootreswls[it, nvar:] = res.bse

print('Bootstrap Results of parameters and parameter standard deviation',)
print('WLS removed 2 outliers from sample')
print('Parameter estimates')
print('median', np.median(bootreswls[:, :5], 0))
print('mean  ', np.mean(bootreswls[:, :5], 0))
print('std   ', np.std(bootreswls[:, :5], 0))

print('Standard deviation of parameter estimates')
print('median', np.median(bootreswls[:, 5:], 0))
print('mean  ', np.mean(bootreswls[:, 5:], 0))
print('std   ', np.std(bootreswls[:, 5:], 0))

plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist(bootreswls[:, i], 50)
    plt.title('var%d' % i)
# @savefig wls_bootstrap_rm2.png
plt.figtext(0.5, 0.935,  'WLS rm2 Bootstrap',
            ha='center', color='black', weight='bold', size='large')


# ..plt.show()
# ..plt.close('all')

'''
# ::
#
#    以下没有固定种子的随机变量
#
#    Bootstrap参数的结果和参数标准偏差
#    OLS
#
#    参数估计
#    median [  -3.26216383  228.52546429  -14.57239967   34.27155426 -227.02816597]
#    mean   [  -2.89855173  234.37139359  -14.98726881   27.96375666 -243.18361746]
#    std    [   3.78704907   97.35797802    9.16316538   94.65031973  221.79444244]
#
#   参数估计值的标准偏差
#    median [   5.44701033   81.96921398    7.58642431   80.64906783  200.19167735]
#    mean   [   5.44840542   86.02554883    8.56750041   80.41864084  201.81196849]
#    std    [   1.43425083   29.74806562    4.22063268   19.14973277   55.34848348]
#
#    bootstrap 参数结果和 WLS 参数标准偏差从样本中剔除了2个离群值
#
#    参数估计
#    median [  -3.95876112  137.10419042   -9.29131131   88.40265447  -44.21091869]
#    mean   [  -3.67485724  135.42681207   -8.7499235    89.74703443  -46.38622848]
#    std    [   2.96908679   56.36648967    7.03870751   48.51201918  106.92466097]
#
#    参数估计值的标准偏差
#    median [   2.89349748   59.19454402    6.70583332   45.40987953  119.05241283]
#    mean   [   2.97600894   60.14540249    6.92102065   45.66077486  121.35519673]
#    std    [   0.55378808   11.77831934    1.69289179    7.4911526    23.72821085]
#
#
#
# 结论: 异常值与可能的异方差问题。
# -----------------------------------------------------------------
#
# bootstrap 的结果
#
# * 与 bootstrap 的标准偏差相比，OLS 模型的 bse 低估了参数的标准偏差
# * 原始数据的 OLS 模型的异方差校正的标准误差（以上）与 bootstrap 标准差较为接近
# * 在删除2个离群值的情况下，使用 bse 的均数或中位数与 bootstrap 中参数估计值的标准差之间匹配的相对较好
#   我们可以在 bootstrap 中包括rsquared，并且也可以为RLM这样做。但这还可能违反了线性假设，例如，尝试对 exog 变量进行非线性变换，但对参数进行线性变换。
#
#
#
# for statsmodels
#
# * 在这种情况下，原始数据的统计模型的 r_squared 看起来较少随机/任意。
# * WLS 模型中估计的r_squared，如果原始exog包含r的平方，请不要将r平方的定义从中心tss更改为非中心tss。
#   一个常数。 由于定义更改而导致的rsquared的增加将极具误导性。
# * 转换后的exog（是否为wexog）是否存在常数项，可能也会影响自由度的计算，但我尚未对此进行检查。我猜 df_model 应该保持不变，但需要使用进行验证。
# * 如果原始数据没有常数项，则必须调整 df_model。在没有常数项的单个 exog变量 进行回归估计 endog变量时，在这种情况下可能需要使用未去中心化的tss来重新定义回归方差分析的 r_squared 和 F检验统计量
#   这可以通过对model.__ init__ 的关键字参数或通过 hasconst =（exog.var（0）<1e-10）.any（） 来自动检测完成。 但我不确定具有完整哑变量组但没有常数项的固定效果。
#  在这种情况下，自动检测将无法以这种方式起作用。 另外，我不确定 ddof 关键字参数是否也可以处理hasconst情况。
'''  # noqa:E501
