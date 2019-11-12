from statsmodels.datasets.macrodata import load_pandas
from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
plt.interactive(False)

# 让我们以 CPI 的 ARIMA 模型来举例 

cpi = load_pandas().data['cpi']
dates = dates_from_range('1959q1', '2009q3')
cpi.index = dates

res = ARIMA(cpi, (1, 1, 1), freq='Q').fit()
print(res.summary())

# 我们可以画图查看序列
cpi.diff().plot()

# 或许查看日志会更好
log_cpi = np.log(cpi)

# 检查 ACF 和 PCF 图
acf, confint_acf = sm.tsa.acf(log_cpi.diff().values[1:], confint=95)
# 将置信区间定为零
# TODO: demean? --> confint_acf -= confint_acf.mean(1)[:, None]
pacf = sm.tsa.pacf(log_cpi.diff().values[1:], method='ols')
# 置信区间是 pacf 的一个选项
confint_pacf = stats.norm.ppf(1 - .025) * np.sqrt(1 / 202.)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_title('Autocorrelation')
ax.plot(range(41), acf, 'bo', markersize=5)
ax.vlines(range(41), 0, acf)
ax.fill_between(range(41), confint_acf[:, 0], confint_acf[:, 1], alpha=.25)
fig.tight_layout()
ax = fig.add_subplot(122, sharey=ax)
ax.vlines(range(41), 0, pacf)
ax.plot(range(41), pacf, 'bo', markersize=5)
ax.fill_between(range(41), -confint_pacf, confint_pacf, alpha=.25)


# TODO: 当 tsa-plots 处于 master时，你可以调整它
# sm.graphics.acf_plot(x, nlags=40)
# sm.graphics.pacf_plot(x, nlags=40)


# 还是有一些季节性
# 尝试使用 ma(4) 项的 arma(1, 1)模型 
