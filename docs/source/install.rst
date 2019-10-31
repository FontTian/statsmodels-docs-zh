:orphan:

.. _install:

安装 statsmodels
======================

安装 statsmodels 的最简单方法是将其安装 `Anaconda <https://docs.continuum.io/anaconda/>`_
发行版一同安装, 后者是用于数据分析和科学计算的跨平台发行版。对于大多数用户，这是推荐的安装方法。

Instructions for installing from PyPI, source or a development version are also provided.


Python 支持
--------------

statsmodels 支持 Python 3.5, 3.6 and 3.7.

Anaconda
--------
statsmodels i可通过
`Anaconda <https://www.continuum.io/downloads>`__ 提供的 conda 来获得，可以使用一下方法安装最新版本：

.. code-block:: bash

   conda install -c conda-forge statsmodels

PyPI (pip)
----------

也可以使用 pip 来安装最新发布的 statsmodels 版本:

.. code-block:: bash

    pip install statsmodels

点击 `this link to our PyPI page <https://pypi.org/project/statsmodels/>`__ 直接下载源代码

对于 Windows 用户, 此处会提供一些非官方的最新文件 `here <https://www.lfd.uci.edu/~gohlke/pythonlibs/#statsmodels>`__.

获取源
--------------------

我们不会经常发布，但是源代码仓库中的 master 分支通常适合日常使用。您可以从我们的
`github repository <https://github.com/statsmodels/statsmodels>`__ 获取最新的资源。 或者，你已经安装了 git:

.. code-block:: bash

    git clone git://github.com/statsmodels/statsmodels.git

如果您想定期了解github上的源代码，请定期执行以下操作:

.. code-block:: bash

    git pull

在 statsmodels 目录中.

Installation from Source
------------------------

您将需要安装C编译器来构建 statsmodels 。如果您是从 github 源而不是源发行版构建的，那么您还将需要
Cython. 您可以按照以下说明获取 Windows 的 C 编译器设置。

如果您的系统已经使用 pip, 编译器 和 git 进行设置，则可以尝试:

.. code-block:: bash

    pip install git+https://github.com/statsmodels/statsmodels

如果您没有安装 pip 或想要手动进行更多安装，则还可以输入：

.. code-block:: bash

    python setup.py install

甚至更多手动

.. code-block:: bash

    python setup.py build
    python setup.py install

statsmodels 也可以在 `develop` 模型下安装，该模式将 statsmodels 直接安装到当前的
 python 环境中。这样做的好处是，当 python 编译器重新启动时，将立即重新编译已编译的模块，
 而无需重新安装 statsmodels 模块。

.. code-block:: bash

    python setup.py develop

编译器
~~~~~~~~~

Linux
^^^^^

如果您使用的是Linux，并且你足够聪明可以自行安装 `gcc` ，尽管它很有可能已经安装好了。

Windows
^^^^^^^

强烈建议你使用 64-bit Python 。

对于Windows用户而言，获得正确的编译器尤其令人困惑。随着时间的流逝，Python已使用各种不同的 Windows C 编译器来构建。
`本指南 <https://wiki.python.org/moin/WindowsCompilers>`_ 有助于阐明默认情况下使用哪个 Python 版本编译器。

Mac
^^^

在 MacOS 上安装 statsmodels 需要安装 `gcc` 它提供了合适的C编译器。我们建议安装 Xcode 和命令行工具。

Dependencies
------------

当前的 minimum dependencies 是:

* `Python <https://www.python.org>`__ >= 3.5
* `NumPy <https://www.scipy.org/>`__ >= 1.14
* `SciPy <https://www.scipy.org/>`__ >= 1.0
* `Pandas <https://pandas.pydata.org/>`__ >= 0.21
* `Patsy <https://patsy.readthedocs.io/en/latest/>`__ >= 0.5.0

需要 Cython 从 git checkout 进行构建，而不是从 PyPI 运行或安装:

* 需要 `Cython <https://cython.org/>`__ >= 0.29 才能从 github 而非源代码分发构建代码。

考虑到较长的发布周期, statsmodels 遵循基于时间的宽松策略来进行 dependencies:
 minimal dependencies 被滞后了大约一年半到两年. 我们的下一个最低版本计划更新预计将在 2020 年上半年。

可选 Dependencies
---------------------

* `cvxopt <https://cvxopt.org/>`__ 是某些模型的常规拟合所必​​需的。
* `Matplotlib <https://matplotlib.org/>`__ >= 2.2 是绘制函数和运行许多示例所需的。
*  如果安装了 `X-12-ARIMA <https://www.census.gov/srd/www/x13as/>`__ 或
  `X-13ARIMA-SEATS <https://www.census.gov/srd/www/x13as/>`__ 则可以用于时间序列分析。
* `pytest <https://docs.pytest.org/en/latest/>`__ 是运行测试套件所必需的
*  如果要在本地构建文档或使用笔记本，需要`IPython <https://ipython.org>`__ >= 5.0 。
* `joblib <http://pythonhosted.org/joblib/>`__ >= 0.9 可用于加速某些模型的分布式估计。
*  需要`jupyter <https://jupyter.org/>`__ 才能运行 notebooks 。
