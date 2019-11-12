"""
自回归移动平均的 (ARMA) 模型
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 从一个 ARMA 程序中生成一些数据
from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(12345)
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])

# arma_generate 函数要求我们为 AR 和 MA 参数指定一个 1，并且 AR 参数为负 

ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]
nobs = 250
y = arma_generate_sample(ar, ma, nobs)

# 现在，可选地，我们可以添加一些日期信息。 在此示例中，我们将使用一个 pandas 的时间序列。
dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = pd.Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2, 2))
arma_res = arma_mod.fit(trend='nc', disp=-1)
