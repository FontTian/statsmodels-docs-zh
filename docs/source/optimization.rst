.. module:: statsmodels.base.optimizer
.. currentmodule:: statsmodels.base.optimizer

优化
============

statsmodels 使用三种类型算法模型的参数估计

  1. 使用适当的线性代数可以直接估算基本线性模型，例如 :ref:`WLS and OLS <regression>` 
  2. :ref:`RLM <rlm>` and :ref:`GLM <glm>`, 使用迭代重新加权的最小二乘。但是，您可以选择选择以下讨论的 scipy 优化器之一
  3. 对于所有其他模型，我们使用 来自 `scipy <https://docs.scipy.org/doc/scipy/reference/index.html>`_ 的
     `optimizers <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ 。


如果可行的话，某些模型允许一个 scipy 优化器的可选项，默认是特定的 scipy 优化器，也可能是一个选项。根据模型和数据，
选择适当的 scipy 优化器可以避免出现局部最小值，可以在更短的时间内拟合模型，或者在内存较少的情况下拟合模型。

statsmodels 支持特定优化器相关的关键字参数大致如下:

- ``newton`` - Newton-Raphson 迭代. 虽然不是直接来自 scipy, 但我们认为它是一种优化程序，因为它仅需要 score 和 hessian 。

    tol : float
        收敛可接受参数的相对误差。

- ``nm`` - scipy's ``fmin_nm``

    xtol : float
        收敛可接受参数的相对误差
    ftol : float
        loglike(params) 中的相对误差可以收敛
    maxfun : int
        function 评估的最大次数

- ``bfgs`` - Broyden–Fletcher–Goldfarb–Shanno optimization, scipy's
  ``fmin_bfgs``.

      gtol : float
          当梯度范数小于 gtol 时停止.
      norm : float
          范数的秩 (np.Inf 是最大, -np.Inf 是最小)
      epsilon
          如果近似 fprime，则将此值用作步长。仅在 LikelihoodModel.score 为 None 时才相关

- ``lbfgs`` - A more memory-efficient (limited memory) implementation of
  ``bfgs``. Scipy's ``fmin_l_bfgs_b``.

      m : int
          用于定义有限存储矩阵的最大可变度量校正数， (有限内存 BFGS
          方法不存储完整的 hessian，而是近似使用此项。.)
      pgtol : float
          当``max{|proj g_i | i = 1, ..., n} <= pgtol`` 时，其中 pg_i 是投影渐变的
           i-th 分量，迭代停止。
      factr : float
          当  ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps`` 时停止迭代，
          其中 eps 代表机器精度, 由代码自动生成. factr的典型值为 : 1e12 （低精度）
          ; 1e7 代表精度适中; 10.0 代表极高的精度。请参阅注释，以了解它与 ftol 的关系, 
          由 L-BFGS-B 的 scipy.optimize.minimize 接口暴露(替代 factr) 。
         
      maxfun : int
          最大迭代次数.
      epsilon : float
          当 approx_grad 为 True 时使用的步长,用于数值计算的梯度。

      approx_grad : bool
          是否近似梯度值 (在这种情况下，func 仅返回函数值)。

- ``cg`` - 共轭梯度优化。 Scipy 的 ``fmin_cg``.

      gtol : float
          当梯度范数小于 gtol 时停止
      norm : float
          范数的秩 (np.Inf 是最大, -np.Inf 是最小)
      epsilon : float
           如果近似 fprime，则将此值用作步长。可是是标量或向量，仅在 LikelihoodModel.score 为 None 时才相关

- ``ncg`` - Newton 共轭梯度。 Scipy 的 ``fmin_ncg``.

      fhess_p : callable f'(x, \*args)
          计算f乘以任意向量p的Hessian的函数。仅当 LikelihoodModel.hessian 为 None 时才提供
      avextol : float
          当最小化器中的平均相对误差低于此值时停止
      epsilon : float or ndarray
          如果近似于 fhess ，则将此值用作步长。仅在 Likelihoodmodel.hessian 为 None 时才相关。

- ``powell`` - Powell's method. Scipy's ``fmin_powell``.

      xtol : float
          Line-search 误差度
      ftol : float
          loglike(params) 中的相对误差可被收敛。
          convergence.
      maxfun : int
          函数评估的最大次数
      start_direc : ndarray
          初始方向设定

- ``basinhopping`` - Basin hopping. 这是 scipy 的 ``basinhopping`` 工具的一部分。

      niter : integer
          basin hopping 的迭代次数
      niter_success : integer
          当全局最小候选数与迭代次数相同时停止迭代
      T : float
          接受或拒绝标准的 "temperature" 参数。较高的 "temperature" 意味着可以接受功能值的较大跳跃。
          为了获得最佳结果，T应该与局部最小值之间的间隔 (函数值) 相当。
      stepsize : float
          用于随机位移的初始步长。
      interval : integer
          更新 `stepsize` 的大小
      minimizer : dict
          传递给最小化器 `scipy.optimize.minimize()`  的额外关键字参数，例如 'method' - 最小化方法
           (例如， 'L-BFGS-B'), 或 'tol' - 终止公差。其他参数是从 `fit` 参数映射而来的 :
          - `args` <- `fargs`
          - `jac` <- `score`
          - `hess` <- `hess`

- ``minimize`` - 允许使用任何 scipy 优化器。

  min_method : str, optional
      使用的最小化方法名称。任何方法特定的参数都可以直接传递。有关方法及其参数的列表，请参见 `scipy.optimize.minimize`.
      的文档。如果未指定任何方法，则使用 BFGS 。

模型类
-----------

通常，最终用户无需直接调用这些函数和类。但是，我们提供此类是因为不同的优化技术具有唯一的关键字参数，这些参数可能对用户有用。

.. autosummary::
   :toctree: generated/

   Optimizer
   _fit_newton
   _fit_bfgs
   _fit_lbfgs
   _fit_nm
   _fit_cg
   _fit_ncg
   _fit_powell
   _fit_basinhopping
