.. currentmodule:: statsmodels.iolib

.. _iolib:

输入-输出 :mod:`iolib`
=========================

``statsmodels`` 提供一些输入和输出功能。其中包括一个用于读取 STATA 文件的阅读器，一个用于生成以多种格式打印的表格的类以及两个用于腌制的辅助功能。

用户还可以利用 :ref:`pandas.io <pandas:io>`. 提供的强大的输入/输出功能。此外, ``pandas`` (a ``statsmodels`` dependency) 还允许
对 Excel、 CSV 和 HDF5 (PyTables) 进行读写。

例子
--------

    `SimpleTable: 基本示例 <examples/notebooks/generated/wls.html#ols-vs-wls>`__

模块参考
----------------

.. module:: statsmodels.iolib
   :synopsis: 用于读取数据集并输出结果的工具

.. autosummary::
   :toctree: generated/

   foreign.StataReader
   foreign.StataWriter
   foreign.genfromdta
   foreign.savetxt
   table.SimpleTable
   table.csv2st
   smpickle.save_pickle
   smpickle.load_pickle


以下是用于返回估计结果的摘要类和函数，而且大多用于内部使用。当前有两个用于创建摘要的版本。

.. autosummary::
   :toctree: generated/

   summary.Summary
   summary2.Summary
