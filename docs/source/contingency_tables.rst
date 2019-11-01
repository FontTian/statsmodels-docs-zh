.. currentmodule:: statsmodels.stats.contingency_tables

.. _contingency_tables:


列联表
==================

statsmodels 支持多种分析列联表的方法，包括评估独立性，对称性，同质性
的方法以及处理来自分层总体的表集合的方法。

这里描述的方法主要用于双向表。可以使用对数线性模型分析多路表。statsmodels当前
没有用于对数线性建模的专用API，但是 :class:`statsmodels.genmod.GLM` 类中的 Poisson 回归可用于此目的。 

列联表是描述数据集的多向表，其中每个观察值属于多个变量中的每个变量的一个类别
例如，如果有两个变量，一个具有 :math:`r` 级别，一个具有 :math:`c` 级别，
则我们有一个 :math:`r \times c` 列联表。可以根据落入表的给定单元格中的观察次数来描述该表，
例如，观测值 :math:`T_{ij}` 表示第一个变量的水平为 :math:`i` ，第二个变量的水平为 :math:`j` 。
请注意，每个变量必须具有有限数量的级别(或类别),可以有序或无序。 在不同的上下文中，定义列联表
的轴的变量可以称为 **categorical variables（分类变量）** or **factor variables（因子变量）**.
它们可以是 **nominal（名义的）** (如果它们的水平是无序的) 或者 **ordinal（有序的）** (如果它们的水平是有序的).

列联表的基础人口由 **分布表** :math:`P_{i, j}` 来描述，其中 :math:`P` 表示概率，所有元素之和的概率 :math:`P` 是 1.
分析列联表的方法是使用 :math:`T` 的数据来了解 :math:`P` 的属性

 :class:`statsmodels.stats.Table` 类是使用列联表最基础的类，我们可以直接通过任何包含列联表单元格计算的矩形数组对象来
 创建一个 Table 对象。

.. ipython:: python

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    df = sm.datasets.get_rdataset("Arthritis", "vcd").data

    tab = pd.crosstab(df['Treatment'], df['Improved'])
    tab = tab.loc[:, ["None", "Some", "Marked"]]
    table = sm.stats.Table(tab)

另外，我们可以传递原始数据，并让 Table 类为我们构造单元格计数数组：

.. ipython:: python

    data = df[["Treatment", "Improved"]]
    table = sm.stats.Table.from_data(data)


Independence（独立性）
------------

**Independence（独立性）** 是行和列因子独立发生的属性。 **Association（联合）** 是缺乏独立性。 
如果联合分布是独立的，则可以将其写为行和列边际分布的外积：

.. math::

    P_{ij} = \sum_k P_{ij} \cdot \sum_k P_{kj} \quad \text{for all} \quad  i, j

我们可以从观测数据中获得最佳拟合的独立分布，然后查看并识别出强烈违反独立性的特定单元格的残差：

.. ipython:: python

    print(table.table_orig)
    print(table.fittedvalues)
    print(table.resid_pearson)

在这个示例中, 与行和列独立的总体样本相比，我们在安慰剂/无改善和治疗/显着改善细胞中观测到太量的观测结果，
而在安慰剂/显着改善和治疗/没有改善细胞中仅观测到很少的结果。这反映出该治疗的明显益处。

如果表的行和列是无序的（即名义变量），那么我们评估独立性的最常用方法是 Pearson's :math:`\chi^2` 统计。
通过查看细胞贡献的 :math:`\chi^2` 卡方统计量，以此作为独立性验证的依据通常是有效的。

.. ipython:: python

    rslt = table.test_nominal_association()
    print(rslt.pvalue)
    print(table.chi2_contribs)

对于有序行和列因子的表，我们可以通过 **linear by linear** 相关检验来获得更多的权重，用来对抗关于排序的替代假设。
线性关联检验的检验统计量是：

.. math::

    \sum_k r_i c_j T_{ij}

其中 :math:`r_i` 和 :math:`c_j` 是行和列的得分，通常这些分数设置为序列 0, 1, ... ， 还给出了 'Cochran-Armitage 趋势检验'。

.. ipython:: python

    rslt = table.test_ordinal_association()
    print(rslt.pvalue)

我们可以通过构造一系列 :math:`2\times 2` 表并计算他们的 odds ratios（比值比）来评估 :math:`r\times x` 表中的关联。有两种方法
可以做到这点，从相邻行和列类别的 **local odds ratios** （本地优势比）来构建 :math:`2\times 2` 表


.. ipython:: python

    print(table.local_oddsratios)
    taloc = sm.stats.Table2x2(np.asarray([[7, 29], [21, 13]]))
    print(taloc.oddsratio)
    taloc = sm.stats.Table2x2(np.asarray([[29, 7], [13, 7]]))
    print(taloc.oddsratio)

也可以通过在每个可能的点上对行和列因子进行二分法的 **cumulative odds ratios** （累积比值比）来构建 :math:`2\times 2` 表。


.. ipython:: python

    print(table.cumulative_oddsratios)
    tab1 = np.asarray([[7, 29 + 7], [21, 13 + 7]])
    tacum = sm.stats.Table2x2(tab1)
    print(tacum.oddsratio)
    tab1 = np.asarray([[7 + 29, 7], [21 + 13, 7]])
    tacum = sm.stats.Table2x2(tab1)
    print(tacum.oddsratio)

 mosaic plot（马赛克图）是一种非正式评估双向表中依赖性的图形方法。

.. ipython:: python

    from statsmodels.graphics.mosaicplot import mosaic
    fig, _ = mosaic(data, index=["Treatment", "Improved"])


对称性和同质性
------------------------

**Symmetry（对称性）** 的属性是 :math:`P_{i, j} = P_{j, i}` 对于每个 :math:`i` 和 :math:`j` 。  
**Homogeneity（同质性）** 是行因子和列因子的边际分布相同的特性

.. math::

    \sum_j P_{ij} = \sum_j P_{ji} \forall i

请注意，要使这些属性适用，表 :math:`P`(和 :math:`T`) 必须为正方形，行和列类别必须相同，并且必须
以相同的顺序出现。

为了说明这点，我们加载一个数据集，创建一个列联表，并计算行和列边距。 :class:`Table` 类包含分析方法 :math:`r \times c` 列联表
下面加载的数据集包含人们左眼和右眼视敏度的评估。我们首先加载数据并创建一个列联表。

.. ipython:: python

    df = sm.datasets.get_rdataset("VisualAcuity", "vcd").data
    df = df.loc[df.gender == "female", :]
    tab = df.set_index(['left', 'right'])
    del tab["gender"]
    tab = tab.unstack()
    tab.columns = tab.columns.get_level_values(1)
    print(tab)

从列联表创建一个 :class:`SquareTable` 对象

.. ipython:: python

    sqtab = sm.stats.SquareTable(tab)
    row, col = sqtab.marginal_probabilities
    print(row)
    print(col)


这种 ``summary`` 方法输出的是对称性和均匀性的检测结果

.. ipython:: python

    print(sqtab.summary())

如果在名为 ``data`` 的 dataframe 中有单独的案例记录，我们同样也可以通过使用 ``SquareTable.from_data`` 类来完成相同的分析。
::

    sqtab = sm.stats.SquareTable.from_data(data[['left', 'right']])
    print(sqtab.summary())


单个2x2表
------------------

 :class:`sm.stats.Table2x2` 类提供了几种处理单个。 ``summary`` 方法显示表的行和列之间的若干关联度量。

.. ipython:: python

    table = np.asarray([[35, 21], [25, 58]])
    t22 = sm.stats.Table2x2(table)
    print(t22.summary())

Note that the risk ratio is not symmetric so different results will be
obtained if the transposed table is analyzed.

.. ipython:: python

    table = np.asarray([[35, 21], [25, 58]])
    t22 = sm.stats.Table2x2(table.T)
    print(t22.summary())


分层2x2表
---------------------

当我们有一组由同样行和列因子定义的列联表时，就会发生分层。在下面的示例中，我们有一组 2x2 表，
反映了中国几个地区吸烟和肺癌的联合分布。表格可能都具有共同的odds ratio（比值比），即使边际概率
在各阶层之间变化也是如此。'Breslow-Day' 程序测试数据是否与常见优势比一致。它在下面显示为
 `Test of constant OR（常数 OR 测试）`。 Mantel-Haenszel 程序测试这个常见优势比是否等于 1 。
它在下面显示为 ` OR=1 的测试`。还可以估计共同的几率和风险比并获得它们的置信区间。
 ``summary``方法显示所有这些结果。可以从类方法和属性中获得单个结果。

.. ipython:: python

    data = sm.datasets.china_smoking.load_pandas()

    mat = np.asarray(data.data)
    tables = [np.reshape(x.tolist(), (2, 2)) for x in mat]

    st = sm.stats.StratifiedTable(tables)
    print(st.summary())


模块参考
----------------

.. module:: statsmodels.stats.contingency_tables
   :synopsis: Contingency table analysis

.. currentmodule:: statsmodels.stats.contingency_tables

.. autosummary::
   :toctree: generated/

   Table
   Table2x2
   SquareTable
   StratifiedTable
   mcnemar
   cochrans_q

也可以看看
--------

Scipy_ 有几个分析列联表的功能，包括 Fisher 的精确测试， 目前在 statsmodels 中还没有。

.. _Scipy: https://docs.scipy.org/doc/scipy-0.18.0/reference/stats.html#contingency-table-functions
