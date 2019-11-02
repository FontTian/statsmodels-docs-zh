:orphan:

.. _missing_data:

缺失数据
------------
所有模型都可以处理丢失的数据。出于性能原因，默认设置是不对丢失的数据进行任何检查。但是，如果您希望内部处理丢失的数据，
则可以使用 missing 关键字参数来实现。默认是什么都不做。

.. ipython:: python

   import statsmodels.api as sm
   data = sm.datasets.longley.load(as_pandas=False)
   data.exog = sm.add_constant(data.exog)
   # 添加一些缺失数据
   missing_idx = np.array([False] * len(data.endog))
   missing_idx[[4, 10, 15]] = True
   data.endog[missing_idx] = np.nan
   ols_model = sm.OLS(data.endog, data.exog)
   ols_fit = ols_model.fit()
   print(ols_fit.params)

默认失败和所有模型参数均为NaN, 这可能不是您所期望的。如果不确定是否存在缺失数据，可以使用 `missing = 'raise'` 。
如果存在缺失数据，将在模型实例化期间引发 `MissingDataError` 以便让你知道输入的数据有问题。

.. ipython:: python
   :okexcept:

   ols_model = sm.OLS(data.endog, data.exog, missing='raise')

如果你希望 statsmodels 处理缺失数据时直接丢弃缺失的观测值，请使用`missing = 'drop'`.

.. ipython:: python

   ols_model = sm.OLS(data.endog, data.exog, missing='drop')

我们正在考虑添加配置框架，以便您可以使用全局设置来设置选项。
