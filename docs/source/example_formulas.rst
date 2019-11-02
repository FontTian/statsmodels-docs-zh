.. _formula_examples:

使用 R-style 公式拟合模型
=====================================

从 0.5.0 版开始 ``statsmodels`` 允许用户使用 R-style 公式拟合统计模型. 在内部, ``statsmodels`` 使用
`patsy <https://patsy.readthedocs.io/en/latest/>`_ 包将公式和数据转换为模型拟合中使用的矩阵。公式框架
非常强大，本教程仅涉及浅显的知识点。可以在 ``patsy`` 文档中找到有关公式语言的完整说明：

-  `Patsy formula language description <https://patsy.readthedocs.io/en/latest/>`_

加载模块和功能
-----------------------------

.. ipython:: python

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np
    import pandas

请注意 ``statsmodels.formula.api`` 除了通常的调用之外，我们还进行了 调用``statsmodels.api`` 。
实际上， ``statsmodels.api`` 仅用于加载数据集， ``formula.api`` 具有许多与 ``api`` (e.g. OLS, GLM)中相同的功能
但是对于大多数模型，它也具有小写字母。通常，小写的模型可以进行 ``formula`` 和 ``df`` 验证, 而大写的模型拥有
``endog`` 和 ``exog`` 设计矩阵. ``formula`` 接受用一个 ``patsy`` 公式描述模型的字符串. ``df`` 需要一个
 `pandas <https://pandas.pydata.org/>`_ 数据框。

``dir(smf)`` 输出一系列的模型

兼容公式的模型具有以下通用呼叫签名:
``(formula, data, subset=None, *args, **kwargs)``

使用公式进行 OLS 回归
-----------------------------

首先，我们拟合 `Getting
Started <gettingstarted.html>`_页面上描述的线性模型。下载数据，子集列和按列删除缺失的观测值：

.. ipython:: python

    df = sm.datasets.get_rdataset("Guerry", "HistData").data
    df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
    df.head()

拟合模型:

.. ipython:: python

    mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
    res = mod.fit()
    print(res.summary())

分类变量
---------------------

查看上面 summary 的输出, 请注意 ``patsy`` 确定 *Region* 特征是文本字符串, 因此将 *Region* 视为
分类变量. ``patsy`` 的默认设置还包括截距，因此我们自动删除了 *Region* 类别之一

如果 *Region* 是我们想要明确地视为分类的整数变量，则可以使用 ``C()`` 运算符来实现：

.. ipython:: python

    res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
    print(res.params)


 ``patsy`` 可以找到分类变量函数更高级的功能: `Patsy: Contrast Coding Systems for categorical variables <contrasts.html>`_

Operators
---------
我们可以使用 "~" 将模型的左侧与右侧分开，而 "+" 将新列添加到设计矩阵中。

删除变量
~~~~~~~~~~~~~~~~~~

The "-" 符号可用于删除列/变量，例如，我们可以通过以下方式从模型中删除截距：

.. ipython:: python

    res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1 ', data=df).fit()
    print(res.params)


Multiplicative interactions（交互项）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

":" 将新列与其他两列的乘积一起添加到设计矩阵中。 "\*" 还将包括相乘在一起的各个列

.. ipython:: python

    res1 = smf.ols(formula='Lottery ~ Literacy : Wealth - 1', data=df).fit()
    res2 = smf.ols(formula='Lottery ~ Literacy * Wealth - 1', data=df).fit()
    print(res1.params)
    print(res2.params)


运算符还可以实现许多其他功能。请查阅 `patsy
docs <https://patsy.readthedocs.io/en/latest/formulas.html>`_ 以了解更多信息。

函数
---------

您可以将向量化函数应用于模型中的变量:

.. ipython:: python

    res = smf.ols(formula='Lottery ~ np.log(Literacy)', data=df).fit()
    print(res.params)


定义一个自定义函数:

.. ipython:: python

    def log_plus_1(x):
        return np.log(x) + 1.0

    res = smf.ols(formula='Lottery ~ log_plus_1(Literacy)', data=df).fit()
    print(res.params)

.. _patsy-namespaces:

Namespaces
----------

请注意，以上所有示例都使用调用名称空间来查找要应用的函数。可以通过 ``eval_env`` 关键字
控制使用的名称空间。例如，您可能想要使用 :class:`patsy:patsy.EvalEnvironment`  提供
自定义名称空间，或者您可能希望使用我们通过传递的 "clean" 名称空间,  ``eval_func=-1`` 
默认是使用调用者的名称空间。例如，如果有一个名为 ``C`` 的变量在用户的命名空间或数据结构
传递给 ``patsy``, 并且变量 ``C`` 在公式中被当成分类变量来处理. 这可能会产生（意外）后果，
请参阅 `Patsy API Reference <https://patsy.readthedocs.io/en/latest/API-reference.html>`_ 了解更多信息。

将公式与尚不支持它们的模型一起使用
---------------------------------------------------------

即使给定的 ``statsmodels`` 函数不支持公式，您仍然可以使用 ``patsy``'的公式语言来生成设计矩阵。然后可以将这些矩阵作为
 ``endog`` 和 ``exog`` 参数提供给拟合函数。

生成 ``numpy`` 数组:

.. ipython:: python

    import patsy
    f = 'Lottery ~ Literacy * Wealth'
    y, X = patsy.dmatrices(f, df, return_type='matrix')
    print(y[:5])
    print(X[:5])

``y`` 和 ``X`` 将是  ``patsy.DesignMatrix`` 的 ``numpy.ndarray``子类实例

生成 pandas 数据框:

.. ipython:: python

    f = 'Lottery ~ Literacy * Wealth'
    y, X = patsy.dmatrices(f, df, return_type='dataframe')
    print(y[:5])
    print(X[:5])

.. ipython:: python

    print(sm.OLS(y, X).fit().summary())
