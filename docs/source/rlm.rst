.. currentmodule:: statsmodels.robust


.. _rlm:

稳健的线性模型
====================

稳健的线性模型支持 `范数`_.下列的 M 估计。 

有关命令和参数，请参见 `Module Reference`_ 

例子
--------

.. ipython:: python

    # 加载模块和数据
    import statsmodels.api as sm
    data = sm.datasets.stackloss.load(as_pandas=False)
    data.exog = sm.add_constant(data.exog)

    # 训练模型并输出 summary
    rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    print(rlm_results.params)

更多详细的示例:

* `Robust Models 1 <examples/notebooks/generated/robust_models_0.html>`__
* `Robust Models 2 <examples/notebooks/generated/robust_models_1.html>`__

技术文档
-----------------------

.. toctree::
   :maxdepth: 1

   rlm_techn1

参考文献
^^^^^^^^^^

* PJ Huber. ‘稳健的统计’ John Wiley and Sons, Inc., New York. 1981.
* PJ Huber. 1973, “ 1972年的Wald纪念演讲：稳健的回归：渐进、猜想和蒙特卡洛” 统计年鉴, 1.5, 799-821.
* R Venables, B Ripley. 纽约州斯普林格的“现代应用统计”

模块参考
----------------

.. module:: statsmodels.robust

模型类
^^^^^^^^^^^^^

.. module:: statsmodels.robust.robust_linear_model
.. currentmodule:: statsmodels.robust.robust_linear_model

.. autosummary::
   :toctree: generated/

   RLM

结果类
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   RLMResults

.. _norms:

范数
^^^^^

.. module:: statsmodels.robust.norms
.. currentmodule:: statsmodels.robust.norms

.. autosummary::
   :toctree: generated/

   AndrewWave
   Hampel
   HuberT
   LeastSquares
   RamsayE
   RobustNorm
   TrimmedMean
   TukeyBiweight
   estimate_location


标准化
^^^^^

.. module:: statsmodels.robust.scale
.. currentmodule:: statsmodels.robust.scale

.. autosummary::
   :toctree: generated/

    Huber
    HuberScale
    mad
    hubers_scale
