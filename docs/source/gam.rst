.. currentmodule:: statsmodels.gam.api

.. _gam:

广义可加模型 (GAM)
=================================

广义可加模型允许对广义线性模型中的平滑项进行惩罚估计。

有关命令和参数，请参见 `Module Reference`_ 

例子
--------

例子说明了高斯和泊松回归，其中将分类变量视为线性项
，而两个解释变量的影响则通过惩罚B-splines来捕获。数据来自汽车数据集
https://archive.ics.uci.edu/ml/datasets/automobile
我们可以加载单元测试模块的 DataFrame .

.. ipython:: python

    import statsmodels.api as sm
    from statsmodels.gam.api import GLMGam, BSplines

    # import data
    from statsmodels.gam.tests.test_penalized import df_autos

    # create spline basis for weight and hp
    x_spline = df_autos[['weight', 'hp']]
    bs = BSplines(x_spline, df=[12, 10], degree=[3, 3])

    # penalization weight
    alpha = np.array([21833888.8, 6460.38479])

    gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos,
                                 smoother=bs, alpha=alpha)
    res_bs = gam_bs.fit()
    print(res_bs.summary())

    # plot smooth components
    res_bs.plot_partial(0, cpr=True)
    res_bs.plot_partial(1, cpr=True)

    alpha = np.array([8283989284.5829611, 14628207.58927821])
    gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos,
                                 smoother=bs, alpha=alpha,
                                 family=sm.families.Poisson())
    res_bs = gam_bs.fit()
    print(res_bs.summary())

    # Optimal penalization weights alpha can be obtaine through generalized
    # cross-validation or k-fold cross-validation.
    # The alpha above are from the unit tests against the R mgcv package.

    gam_bs.select_penweight()[0]
    gam_bs.select_penweight_kfold()[0]


参考
^^^^^^^^^^

* Hastie, Trevor, and Robert Tibshirani. 1986. 广义可加模型. Statistical Science 1 (3): 297-310.
* Wood, Simon N. 2006. 广义可加模型: An Introduction with R. 统计科学笔记. Boca Raton, FL: Chapman & Hall/CRC.
* Wood, Simon N. 2017. 广义可加模型: R的简介. 第二版. Chapman & Hall/CRC 的统计科学教科书. Boca Raton: CRC Press/Taylor & Francis Group.


模块参考
----------------

.. module:: statsmodels.gam.generalized_additive_model
   :synopsis: Generalized Additive Models
.. currentmodule:: statsmodels.gam.generalized_additive_model

模型类
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLMGam
   LogitGam

结果类
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLMGamResults

Smooth Basis Functions
^^^^^^^^^^^^^^^^^^^^^^

.. module:: statsmodels.gam.smooth_basis
   :synopsis: Classes for Spline and other Smooth Basis Function

.. currentmodule:: statsmodels.gam.smooth_basis

Currently there is verified support for two spline bases

.. autosummary::
   :toctree: generated/

   BSplines
   CyclicCubicSplines

`statsmodels.gam.smooth_basis` includes additional splines and a (global)
polynomial smoother basis but those have not been verified yet.



家族模型和 Link 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^

GLMGam 与 GLM 相同，具有相同的 link 函数。当前的联合测试仅涵盖了高斯和泊松，
并且GLMGam可能不适用于GLM中所有选项。