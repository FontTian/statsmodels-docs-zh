"""
在时间序列模型中使用日期
"""
import statsmodels.api as sm
import pandas as pd

# 入门
# ---------------

data = sm.datasets.sunspots.load()

# 现在，一个年度日期序列必须是该年末的日期时间。

dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))

# 使用 Pandas
# ------------

# 使用一个具有 DatetimeIndex 的 pandas Series 或 DataFrame
endog = pd.Series(data.endog, index=dates)

# 并实例化模型
ar_model = sm.tsa.AR(endog, freq='A')
pandas_ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)

# 让我们做一些样本外预测
pred = pandas_ar_res.predict(start='2005', end='2015')
print(pred)

# 使用明确的日期
# --------------------

ar_model = sm.tsa.AR(data.endog, dates=dates, freq='A')
ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
pred = ar_res.predict(start='2005', end='2015')
print(pred)

# 由于模型具有日期信息，仅仅返回一个 regular 数组，另外，您可以通过回旋方式获取预测日期。

print(ar_res.data.predict_dates)

# 如果 predict 方法一旦被调用，这个属性才会存在的。它保存了上一次调用关联日期的 predict 方法

# TODO: 应该与结果的实例化对象连接起来?
