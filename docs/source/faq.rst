:orphan:

.. currentmodule:: statsmodels

.. _faq:

常见问题
=========================

statsmodels 是什么?
--------------------

statsmodels 是一个 Python 软件包，提供了广泛使用的统计模型的集合。虽然 statsmodels 过去有大量的计量经济学用户群， 
但该软件包的设计目的是可用于多种统计用例。与其他基于 Python 的建模工具相比， statsmodels 更加侧重于模型的基础统计
和诊断，而不是拥有最前沿的或预测性的模型。

.. _endog-exog-faq:

endog 和 exog 是什么?
----------------------------

这些是 endogenous 和 exogenous 变量的缩写. 可能线性模型表示为通用 ``y`` 和 ``X`` 会让你更满意， 有时将内生变量
 ``y`` 称为因变量。同样有时将外生变量称为自变量，你可以在 :ref:`endog_exog` 了解更多的详细内容。

.. _missing-faq:

statsmodels 是如何处理缺失值?
-----------------------------------------

缺失值可以通过 ``missing`` 关键字参数来处理，每个模型都使用此关键字。
您可以在 :class:`statsmodels.base.Model <statsmodels.base.model.Model>`的文档中找到更多信息 。

.. _build-faq:

为什么不构建 statsmodels ?
-------------------------------

请记住，要构建，您必须具有:

- 安装了适当的依赖项 (numpy, pandas, scipy, Cython) 
- 合适的 C 编译器
- 可以正常工作的 python 安装包

请查看我们的 :ref:`installation instructions <install>` 获得详细信息。

您也可以尝试通过运行以下命令清理源目录:

.. code-block:: bash

    pip uninstall statsmodels
    python setup.py clean

然后尝试重新编译。如果您想更积极一些，也可以通过以下方法将 git 重置为以前的版本：

.. code-block:: bash

    git reset --hard
    git clean -xdf
    git checkout master
    python setup.py clean

如果我想贡献，该从哪里开始?
-----------------------------------------

请查看我们的 :doc:`development pages <dev/index>` 以获取有关如何参与的指南。我们在 GitHub 页面上接受请求请求，获取与统计和统计
建模密切相关的错误修正和主题。此外，人们也非常赞赏可用性和生活质量的提高。

如果我的问题在这里没有回答怎么办?
-----------------------------------------

您可能会在 GitHub 的 `FAQ issues tag <https://github.com/statsmodels/statsmodels/labels/FAQ>`_.
下找到尚未在此处添加的问题的答案。如果没有，请使用
`statsmodels tag <https://stackoverflow.com/questions/tagged/statsmodels>`_ 或
 `mailing list <https://groups.google.com/forum/#!forum/pystatsmodels>`_ 询问有关 stackoverflow 的问题。
