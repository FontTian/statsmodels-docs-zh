.. image:: images/statsmodels-logo-v2-horizontal.svg
   :width: 50%
   :alt: statsmodels
   :align: left

:ref:`statsmodels <about:About statsmodels>` 是一个Python模块，它提供用于估计的许多不同统计模型，
进行统计检验和统计数据探索的类和函数。每个估算器都有大量结果统计信息列表。将结果与现有统计数据包进行测试，
以确保结果正确。该软件包是根据开源的经过修改的 BSD (3-条款) 许可发布.在线文档托管在 `statsmodels.org <https://www.statsmodels.org/>`__.

较少
============

``statsmodels`` 支持使用 R-样式公式和 ``pandas`` DataFrames。这是一个普通最小二乘法的示例:

.. ipython:: python

    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # 加载数据
    dat = sm.datasets.get_rdataset("Guerry", "HistData").data

    # 拟合回归模型 (使用回归变量之一的自然对数)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

    # 输出结果
    print(results.summary())

你也可以使用 ``numpy`` 数组来替代公式:

.. ipython:: python

    import numpy as np
    import statsmodels.api as sm

    # 生成数据 (2 个回归项 + 1 个常数)
    nobs = 100
    X = np.random.random((nobs, 2))
    X = sm.add_constant(X)
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(X, beta) + e

    # 拟合回归模型
    results = sm.OLS(y, X).fit()

    # 输出结果
    print(results.summary())

通过 `dir(results)` 来查看结果. 属性的描述在 `results.__doc__` ，且结果的方法有其自己的文档。

引文
========

在科学出版物中使用统计模型时，请考虑使用以下引用：


Seabold, Skipper, and Josef Perktold. "`使用 python 进行经济计量和统计建模. <http://conference.scipy.org/proceedings/scipy2010/pdfs/seabold.pdf>`_" *Proceedings
第 9 届python科学会议论文集.* 2010.

Bibtex entry::

  @inproceedings{seabold2010statsmodels,
    title={statsmodels: Econometric and statistical modeling with python},
    author={Seabold, Skipper and Perktold, Josef},
    booktitle={9th Python in Science Conference},
    year={2010},
  }

.. toctree::
   :maxdepth: 1

   install
   gettingstarted
   user-guide
   examples/index
   release/index
   api
   about
   dev/index


Index
=====

* :ref:`genindex`
* :ref:`modindex`
