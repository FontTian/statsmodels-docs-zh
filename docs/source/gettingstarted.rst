入门
===============

这个非常简单的案例研究旨在帮助您快速入门 ``statsmodels``. 从原始数据开始，我们将显示估计统计模型和绘制诊断图所需的步骤。
我们仅使用由 ``statsmodels`` 或者与其有以来关系的 ``pandas`` 和 ``patsy`` 提供的函数。

加载模块和函数
-----------------------------

After `installing statsmodels and its dependencies <install.html>`_, we load a
few modules and functions:

.. ipython:: python

    import statsmodels.api as sm
    import pandas
    from patsy import dmatrices

`pandas <https://pandas.pydata.org/>`_ 建立在 ``numpy`` 数组上，可提供丰富的数据结构和数据分析工具。
  ``pandas.DataFrame`` 函数提供带标签的 (可能是异构的) 数据, 类似于
``R`` "data.frame".  ``pandas.read_csv`` 函数可用于以逗号作为分隔符的数据文件转换为 ``DataFrame`` 对象

`patsy <https://github.com/pydata/patsy>`_ 是一个 Python 库，用于描述统计模型和构建 `Design Matrices
<https://en.wikipedia.org/wiki/Design_matrix>`_ 使用 ``R``-风格的公式.

.. 注意::

   本示例使用 API 接口。 有关导入 API 接口与直接从定义的模块中导入信息的区别， 请参考 :ref:`importpaths` 
    (``statsmodels.api`` 和 ``statsmodels.tsa.api``) 

数据集
----

我们加载了 `Guerry dataset
<https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Guerry.html>`_, 该数据集是历史数据的集合，
用于支持 Andre-Michel Guerry's 1833 *Essay on the Moral Statistics of France*. 数据集由 `Rdatasets
<https://github.com/vincentarelbundock/Rdatasets/>`_ 仓库以逗号分隔符（ CSV ）格式在线托管。
在加载本地文件时，我们使用 ``read_csv`` 函数, 而且
``pandas`` 自动为我们处理了所有这些的工作:

.. ipython:: python

    df = sm.datasets.get_rdataset("Guerry", "HistData").data

 在`Input/Output doc page <iolib.html>`_ 页面展示了如何导入其他格式的数据集

我们可以选择感兴趣的变量，并查看底部 5 行:

.. ipython:: python

    vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
    df = df[vars]
    df[-5:]

请注意，在 *Region* 列中缺少一个观测值. 我们可以使用 ``pandas`` 提供的 ``DataFrame`` 方法来解决它： 

.. ipython:: python

    df = df.dropna()
    df[-5:]

Substantive motivation and model
--------------------------------

我们想知道 86 French departments 的识字率是否与 1820s 皇家彩票的人均赌注相关 . 我们需要控制每个部门的财富水平,
还希望在回归方程的右侧包括一系列虚拟变量，以控制由于区域效应而导致的未观测到的差异性。 使用普通最小二乘法 (OLS)
来估算模型

设计矩阵 (*endog* & *exog*)
----------------------------------

为了适合 ``statsmodels`` 所涵盖的大多数模型， 您将需要创建两个设计矩阵，第一个是内生变量 (即因变量、响应变量和回归变量等)。
第二个是外生变量变量 (自变量、 预测变量、 回归变量, 等等.)。
OLS 模型系数估计值照常计算：

.. math::

    \hat{\beta} = (X'X)^{-1} X'y

其中 :math:`y`  是人均彩票投注的 :math:`N \times 1` 列的数据 (*Lottery*)。 :math:`X` 是 :math:`N \times 7` 并带有截距，
*Literacy* 和 *Wealth* 变量, 以及 4 个区域二元变量。

 ``patsy`` 模块提供了使用类似 ``R``-公式来准备设计矩阵的便捷功能. 你可以在此处 `here <https://patsy.readthedocs.io/en/latest/>`_ 找到更多信息。

我们使用 ``patsy`` 的 ``dmatrices`` 函数来创建设计矩阵:

.. ipython:: python

    y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')

生成的矩阵/数据框如下所示：

.. ipython:: python

    y[:3]
    X[:3]

注意 ``dmatrices`` 有

* 将分类变量 *Region* 拆分为一组指标变量.
* 在外生回归矩阵中增加一个常数项
* 返回 ``pandas`` DataFrame 而不是简单的numpy数组。因为 DataFrame 可以携带元数据 (如： 变量名) ，statsmodels 在展示结果就非常的有用。

上述行为也可以更改，请参阅 `patsy doc pages
<https://patsy.readthedocs.io/en/latest/>`_.

模型拟合和 summary 汇总
---------------------

拟合模型 ``statsmodels`` 通常有以下3个简单步骤:

1. 使用模型类来描述模型
2. 使用模型类的方法拟合模型
3. 使用汇总方法检查结果

对于 OLS, 可以通过一下方法来实现:

.. ipython:: python

    mod = sm.OLS(y, X)    # 描述模型
    res = mod.fit()       # 拟合模型
    print(res.summary())   # 汇总模型


 ``res`` 对象有很多有用的的属性。如，我们可以通过一下内容来提取参数估计值和 r 方：

.. ipython:: python

    res.params
    res.rsquared

输入 ``dir(res)`` 可以查看属性的完整列表。

更多信息和示例，请参阅 `Regression doc page <regression.html>`_ 页面

诊断和规范检验
-----------------------------------

``statsmodels`` 可以进行一系列有用的 `regression diagnostics
and specification tests
<stats.html#residual-diagnostics-and-specification-tests>`_.  例如,
对彩虹进行线性检验 (零假设是将关系正确建模为线性):

.. ipython:: python

    sm.stats.linear_rainbow(res)

诚然，上面产生的输出不是很冗长，但是通过阅读 `docstring <generated/statsmodels.stats.diagnostic.linear_rainbow.html>`_
我们知道 (also, ``print(sm.stats.linear_rainbow.__doc__)``) ，第一个数字是 F-统计量，第二个数字是 p-value.

``statsmodels`` 还提供了绘图函数。 例如, 我们可以通过以下方式绘制一组回归变量的回归图：

.. ipython:: python

    @savefig gettingstarted_0.png
    sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],
                                 data=df, obs_labels=False)

文献资料
-------------
可以使用 :func:`~statsmodels.tools.web.webdoc` 从IPython中访问文档

.. autosummary::
   :nosignatures:
   :toctree: generated/

   ~statsmodels.tools.web.webdoc

更多
----

恭喜你! 你已准备好了进入
`Table of Contents <index.html#table-of-contents>`_ 的其他主题
