.. module:: statsmodels.duration
   :synopsis: Models for durations

.. currentmodule:: statsmodels.duration


.. _duration:

生存和持续时间分析方法
==========================================

:mod:`statsmodels.duration` 实现了用于处理审查数据的几种标准方法，这些方法常用于处理事件起始时间点到感兴趣事件发生的时间点之间的数据。
典型的示例是医学研究，其起始是诊断对象患有某种疾病的时间点，感兴趣事件是死亡时间点（或疾病发展、恢复等）。

当前仅处理 right-censoring，当我们知道某个事件在给定时间t之后发生，但我们不知道确切的事件发生的时间时，便会进行right-censoring。

生存函数估计和推断
------------------------------------------

 :class:`statsmodels.api.SurvfuncRight` 类可以被用来估计 right-censoring 数据的生存函数。``SurvfuncRight`` 实现几种推理程序，
 包括生存分布分位数的置信区间，生存函数的点估计和置信区，以及绘图程序。 ``duration.survdiff`` 提供了用于比较生存分布的检验流程。


我们使用从 R 数据集仓库中获得的，来自于 `flchain` 研究的数据集来创建一个 ``SurvfuncRight`` 对象，我们仅对女性受试者的数据进行生存分布拟合。


.. code-block:: python

   import statsmodels.api as sm

   data = sm.datasets.get_rdataset("flchain", "survival").data
   df = data.loc[data.sex == "F", :]
   sf = sm.SurvfuncRight(df["futime"], df["death"])

通过调用 ``summary`` 方法可以看到拟合生存分布的主要特征：

.. code-block:: python

    sf.summary().head()

我们可以获得生存分布的分位数的点估计和置信区间。由于在这项研究中只有约30％的受试者死亡，
因此我们只能估算低于0.3个概率点的分位数：


.. code-block:: python

    sf.quantile(0.25)
    sf.quantile_ci(0.25)

要绘制单个生存函数，请调用 ``plot`` 方法:

.. code-block:: python

    sf.plot()

由于这是一个大量检查的大型数据集，因此希望不用绘制检查标志：

.. code-block:: python

    fig = sf.plot()
    ax = fig.get_axes()[0]
    pt = ax.get_lines()[1]
    pt.set_visible(False)

我们还可以将95％置信区间添加到绘图中。通常，这些区间仅针对分布的中心部分绘制。

.. code-block:: python

    fig = sf.plot()
    lcb, ucb = sf.simultaneous_cb()
    ax = fig.get_axes()[0]
    ax.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')
    ax.set_xlim(365, 365*10)
    ax.set_ylim(0.7, 1)
    ax.set_ylabel("Proportion alive")
    ax.set_xlabel("Days since enrollment")

在这里，我们在同一轴上绘制两组（女性和男性）生存函数:

.. code-block:: python

    gb = data.groupby("sex")
    ax = plt.axes()
    sexes = []
    for g in gb:
        sexes.append(g[0])
        sf = sm.SurvfuncRight(g[1]["futime"], g[1]["death"])
        sf.plot(ax)
    li = ax.get_lines()
    li[1].set_visible(False)
    li[3].set_visible(False)
    plt.figlegend((li[0], li[2]), sexes, "center right")
    plt.ylim(0.6, 1)
    ax.set_ylabel("Proportion alive")
    ax.set_xlabel("Days since enrollment")

我们可以通过 ``survdiff`` 形式比较两个生存分布，它实现了几种标准的非参数过程。默认程序是 logrank 测试:

.. code-block:: python

    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex)

这是通过 survdiff 实现的其他一些测试过程

.. code-block:: python

    # Fleming-Harrington ， p=1, i.e. weight by pooled survival time
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='fh', fh_p=1)

    # Gehan-Breslow, weight by number at risk
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='gb')

    # Tarone-Ware, weight by the square root of the number at risk
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='tw')


回归方法
------------------

比例风险回归模型 ("Cox 模型") 是用于审查数据的回归技术。它们使事件发生时间的变化可以用协变量来解释，
类似于线性或广义线性回归模型中所做的事情。这些模型用 "hazard ratios 风险比" , 表示协变量效应，这意味着根据
协变量的值，将危害 (瞬时事件发生率) 乘以给定因子。


.. code-block:: python

   import statsmodels.api as sm
   import statsmodels.formula.api as smf

   data = sm.datasets.get_rdataset("flchain", "survival").data
   del data["chapter"]
   data = data.dropna()
   data["lam"] = data["lambda"]
   data["female"] = (data["sex"] == "F").astype(int)
   data["year"] = data["sample.yr"] - min(data["sample.yr"])
   status = data["death"].values

   mod = smf.phreg("futime ~ 0 + age + female + creatinine + "
                   "np.sqrt(kappa) + np.sqrt(lam) + year + mgus",
                   data, status=status, ties="efron")
   rslt = mod.fit()
   print(rslt.summary())


更多详细示例，请参见 :ref:`statsmodels-examples` 。


在维基百科有一些示例笔记:
`用于 PHReg 和 生存分析的维基笔记 <https://github.com/statsmodels/statsmodels/wiki/Examples#survival-analysis>`_


.. todo::

   Technical Documentation

参考文献
^^^^^^^^^^

 Cox 比例风险回归模型的参考 ::

    T Therneau (1996). Extending the Cox model. Technical report.
    http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

    G Rodriguez (2005). Non-parametric estimation in survival models.
    http://data.princeton.edu/pop509/NonParametricSurvival.pdf

    B Gillespie (2006). Checking the assumptions in the Cox proportional
    hazards model.
    http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf


模块参考
----------------

.. module:: statsmodels.duration.survfunc
   :synopsis: Models for Survival Analysis

.. currentmodule:: statsmodels.duration.survfunc

用于生存分布的类:

.. autosummary::
   :toctree: generated/

   SurvfuncRight

.. module:: statsmodels.duration.hazard_regression
   :synopsis: Proportional hazards model for Survival Analysis

.. currentmodule:: statsmodels.duration.hazard_regression

比例风险回归模型的类:

.. autosummary::
   :toctree: generated/

   PHReg

比例风险回归结果的类:

.. autosummary::
   :toctree: generated/

   PHRegResults
