.. currentmodule:: statsmodels.genmod.bayes_mixed_glm

广义线性混合效应模型
=======================================

广义线性混合效应（GLIMMIX）模型是在线性预测变量中具有随机效应的广义线性模型。
statsmodel当前支持使用两种贝叶斯方法对二项式和泊松 GLIMMIX模型进行估计
后验的拉普拉斯近似，和后验的可变贝叶斯近似.  两种方法均提供了点估计 (后验均值) 和
不确定性估计 (后验标准偏差).

当前的实现仅支持独立的随机效果.

技术文档
-----------------------

与statsmodels混合线性模型不同，GLIMMIX不是基于组来实现的，而是通过将所有分类变量的随机效应的交互来创建组。
注意，这样就会创建大的、稀疏的随机效应设计矩阵 `exog_vc`.  在内部, `exog_vc` 被转换为稀疏矩阵.  
当把参数直接传递给类初始化程序时，可能也会传递一个稀疏矩阵；使用公式时, 将创建一个密集矩阵，并将其转换为
稀疏矩阵。对于非常大的问题，由于该密集中间矩阵的大小，使用公式可能不可行。

参考文献
^^^^^^^^^^

Blei, Kucukelbir, McAuliffe (2017).  Variational Inference: A review
for Statisticians https://arxiv.org/pdf/1601.00670.pdf

模块参考
----------------

.. module:: statsmodels.genmod.bayes_mixed_glm
   :synopsis: Bayes Mixed Generalized Linear Models


模型类:

.. autosummary::
   :toctree: generated/

   BinomialBayesMixedGLM
   PoissonBayesMixedGLM

结果类:

.. autosummary::
   :toctree: generated/

   BayesMixedGLMResults
