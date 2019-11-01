:orphan:

.. _dataset_proposal:

statsmodels 的数据集: design proposal
===============================================

 numpy/scipy 缺少一系列可用于演示、教程的数据集，例如 R 有一系列的核心数据集。

数据集的用途如下:

        - 示例，模型用法教程
        - 检验模型效果与其他统计信息的数据包

也就是说，数据集不仅是数据，而且还包括一些元数据。本提案的目的是提出一种简单明了且
又不妨碍数据特定用途的组织数据的通用做法。


背景
----------

This proposal was adapted from David Cournapeau's original proposal for a
datasets package for scipy and the learn scikit.  It has been adapted for use
in the statsmodels scikit.  The structure of the datasets itself, while
specific to statsmodels, should be general enough such that it might be used
for other types of data (e.g., in the learn scikit or scipy itself).

组织
------------

Each dataset is a directory in the `datasets` directory and defines a python
package (e.g. has the __init__.py file). Each package is expected to define the
function load, returning the corresponding data. For example, to access datasets
data1, you should be able to do::

  >>> from statsmodels.datasets.data1 import load
  >>> d = load() # -> d is a Dataset object, see below

The `load` function is expected to return the `Dataset` object, which has certain
common attributes that make it readily usable in tests and examples. Load can do
whatever it wants: fetching data from a file (python script, csv file, etc...),
generating random data, downloading from the Internet, etc. The `load` function
will return the data as a pandas DataFrame.

It is strongly recommended that each dataset directory contain a csv file with
the dataset and its variables in the same form as returned by load so that the
dataset can easily be loaded into other statistical packages.  In addition, an
optional (though recommended) sub-directory src should contain the dataset in
its original form if it was "cleaned" (ie., variable transformations) in order
to put it into the format needed for statsmodels. Some special variables must
be defined for each package, containing a Python string:

    - COPYRIGHT: copyright information
    - SOURCE: where the data are coming from
    - DESCHOSRT: short description
    - DESCLONG: long description
    - NOTE: some notes on the datasets.

See `datasets/data_template.py` for more information.

数据格式
------------------

强烈建议使用load函数返回的 `Dataset` 对象，代替使用类提供元数据，而是使用 Bunch 模式。

::

  class Bunch(dict):
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

请参考 `Reference <http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/>`_

在实践中，您可以使用 ::

  >>> from statsmodels.datasets import Dataset

作为默认收集器，如 `datasets/data_template.py`.

Bunch 模式的优点是它保留了按属性查找。主要目标是：

    - 对于只需要数据的人来说，没有额外的负担
    - 对于需要更多内容的人，他们可以轻松地从返回的值中提取所需的内容。
      通过此模型可以轻松构建更高级别的抽象。
    - 所有可能的数据集都应适合此模型

为了使数据集在statsmodels中有用，封装 Dataset 对象应该遵从一下约定和属性：


    - 调用对象本身将返回完整数据集的纯 ndarray 。
    - `data`: 包含实际数据的Recarray。假设此时所有数据都可以安全地转换为浮点数。
    - `raw_data`: 这是'data'的纯ndarray版本。 
    - `names`: 这将返回 data.dtype.names 以便 name[i] 是 'raw_data' 中的第i列。
    - `endog`: t提供此值是为了方便测试和示例
    - `exog`: 提供此值是为了方便测试和示例
    - `endog_name`: endog 属性的名称
    - `exog_name`: exog 属性的名称

它包含足够的信息，可以通过自省和简单功能获得所有有用的信息。此外，可以轻松添加对其他程序包可能有用的属性。

添加数据集
----------------

请参考 :ref:`notes on adding a dataset <add_data>`.

用法示例
-------------

::

  >>> from statsmodels import datasets
  >>> data = datasets.longley.load(as_pandas=True)

仍然存在的问题:
-------------------

    - 如果数据集很大且无法容纳到内存中，我们要避免使用哪种API来将所有数据加载到内存中？
      我们可以使用内存映射数组吗？
    - Missing data: 我曾考虑过将记录数组和掩码数组类都子类化，但是我不知道这是否可行，
      甚至还没有道理。我有一些数据挖掘软件使用Nan的感觉 (例如, weka 似乎在内部使用 float ),
       但这阻止了它们表示整数数据。
    - 如何处理 non-float 数据，即字符串或分类变量？


Current implementation
----------------------

An implementation following the above design is available in `statsmodels`.


注意
----

Although the datasets package emerged from the learn package, we try to keep it
independent from everything else, that is once we agree on the remaining
problems and where the package should go, it can easily be put elsewhere
without too much trouble. If there is interest in re-using the datasets package,
please contact the developers on the `mailing list <https://groups.google.com/forum/?hl=en#!forum/pystatsmodels>`_.
