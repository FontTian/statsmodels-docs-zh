.. _datasets:

.. currentmodule:: statsmodels.datasets

.. ipython:: python
   :suppress:

   import numpy as np
   np.set_printoptions(suppress=True)

数据集
====================

``statsmodels`` 提供用于示例，教程，模型测试等的数据集（即数据和元数据）。

使用 Stata 中的数据集
-------------------------

.. autosummary::
   :toctree: ./

   webuse

使用 R 中的数据集
---------------------

The `Rdatasets project <https://vincentarelbundock.github.io/Rdatasets/>`__ 可以访问R中的核心数据集包和许多其他常见的R程序包提供的数据集。通过使用 :func:`get_rdataset` 函数，所有这些数据集都可用于 statsmodels ， ``data`` 属性可以访问实际数据。例如

.. ipython:: python

   import statsmodels.api as sm
   duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
   print(duncan_prestige.__doc__)
   duncan_prestige.data.head(5)


R 数据集功能参考
-----------------------------


.. autosummary::
   :toctree: ./

   get_rdataset
   get_data_home
   clear_data_home


可用数据集
------------------

.. toctree::
   :maxdepth: 1
   :glob:

   generated/*

用法
-----

加载数据集:

.. ipython:: python

   import statsmodels.api as sm
   data = sm.datasets.longley.load_pandas()

 `Dataset` 对象遵循 :ref:`proposal <dataset_proposal>` 解释的束模式。data 属性中提供了完整的数据集。
.. ipython:: python

   data.data

大多数数据集都在 `endog` 和 `exog`属性中方便地展示数据：

.. ipython:: python

   data.endog.iloc[:5]
   data.exog.iloc[:5,:]

但是，单变量数据集没有 `exog` 属性。

可以通过输入以下内容获取变量名称：

.. ipython:: python

   data.endog_name
   data.exog_name

如果数据集对应为 `endog` 和 `exog`没有明确的解释，则您始终可以访问 `data` 或 `raw_data` 属性。
宏观数据数据集就是这种情况，它是美国宏观经济数据的集合，而不是带有特定示例的数据集。`data` 属性
包含完整数据集的记录数组，而`raw_data` 属性包含 ndarray ，其名称由 `data` 属性提供。

.. ipython:: python

   type(data.data)
   type(data.raw_data)
   data.names

将数据作为 pandas 对象加载
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于许多用户而言，最好将数据集作为 pandas DataFrame 或 Series 对象来获取。每个数据集模块都有
一种 ``load_pandas`` 方法，该方法返回一个 ``Dataset`` 实例，该实例具有作为熊猫对象随时可用的数据：

.. ipython:: python

   data = sm.datasets.longley.load_pandas()
   data.exog
   data.endog

完整的 DataFrame ， ``data`` 在 Dataset 对象的属性中可用

.. ipython:: python

   data.data


通过将熊猫集成到估计类中，元数据将附加到模型结果中：

.. ipython:: python
   :okwarning:

   y, x = data.endog, data.exog
   res = sm.OLS(y, x).fit()
   res.params
   res.summary()

其他信息
^^^^^^^^^^^^^^^^^

如果您想了解有关数据集本身的更多信息，可以再次使用 Longley 数据集为例，访问以下内容 ::

    >>> dir(sm.datasets.longley)[:6]
    ['COPYRIGHT', 'DESCRLONG', 'DESCRSHORT', 'NOTE', 'SOURCE', 'TITLE']

附加信息
----------------------

* 数据集软件包的想法最初是由 David Cournapeau 提出的，可以在 :ref:`here <dataset_proposal>` 并由 Skipper Seabold 进行更新。
* 要添加数据集，请参阅 :ref:`notes on adding a dataset <add_data>`.
