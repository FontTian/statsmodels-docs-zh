.. module:: statsmodels
   :synopsis: Statistical analysis in Python

.. currentmodule:: statsmodels

*****************
关于statsmodels
*****************

背景
----------

模块最初由乔纳森·泰勒（Jonathan Taylor）编写的 scipy.stats。一直以来，它是 scipy 的一部分，但后来被删除了。
在 Google Summer of Code 2009 期间，我们对其 statsmodels 进行了更正、测试、改进和发布，并将其作为新程序包发布。
从那时起，statsmodels 开发团队就继续添加新模型，绘图工具和统计方法。

测试
-------

大多数结果已通过至少一个其他统计软件包：R，Stata 或 SAS 进行了验证。初始重写和持续开发的指导原则是必须验证所有数字。
一些统计方法已通过蒙特卡洛研究进行了检验。尽管我们努力遵循这种测试驱动的方法，但不能保证代码没有错误，并且始终有效。
某些辅助功能仍未充分测试，某些边缘情况可能未正确考虑，并且许多统计模型都固有有数字问题的可能性。
我们特别感谢您提供的有关此类问题的帮助和报告，以便我们可以不断改进现有模型。


Code 稳定性
^^^^^^^^^^^^^^

现有模型大部分都放置在其用户界面中，并且我们预计以后不会有很多重大变化。对于现有代码，尽管尚不能保证API的稳定性，
但在非常特殊的情况下，除所有特殊情况外，我们都有较长的弃用期，并且我们尝试将需要现有用户进行调整的更改保持在最低水平。
对于较新的模型，我们可能会在获得更多经验并获得反馈时调整用户界面。这些更改将始终在文档中的发行说明中记录。


Bugs 反馈
^^^^^^^^^^^^^^
If you encounter a bug or an unexpected behavior, please report it on
`the issue tracker <https://github.com/statsmodels/statsmodels/issues>`_.
Use the ``show_versions`` command to list the installed versions of
statsmodels and its dependencies.

.. autosummary::
   :toctree: generated/

   ~statsmodels.tools.print_version.show_versions


经济支持
-----------------

我们感谢为 statsmodels 提供的经济支持：

* Google `www.google.com <https://www.google.com/>`_ : Google Summer of Code
  (GSOC) 2009-2017.
* AQR `www.aqr.com <https://www.aqr.com/>`_ : 从事矢量自回归模型（VAR）工作的财务赞助商

我们还要感谢托管服务提供商, `github
<https://github.com/>`_ 提供了公共代码存储库, `github.io
<https://www.statsmodels.org/stable/index.html>`_ 托管了我们的文档
and `python.org <https://www.python.org/>`_ for making our downloads available
on PyPi.

我们还要感谢我们的持续整合提供商,`Travis CI <https://travis-ci.org/>`_ 和 `AppVeyor <https://ci.appveyor.com>`_ 为单元测试，
以及 `Codecov <https://codecov.io>`_ 和 `Coveralls <https://coveralls.io>`_ 为代码覆盖率

商标品牌
-----------

在做关于 statsmodels 代码的演示时，请使用 statsmodels 的 loge

彩色
^^^^^

+----------------+---------------------+
| Horizontal     | |color-horizontal|  |
+----------------+---------------------+
| Vertical       | |color-vertical|    |
+----------------+---------------------+
| Logo Only      | |color-notext|      |
+----------------+---------------------+

单色 (Dark)
^^^^^^^^^^^^^^^^^

+----------------+---------------------+
| Horizontal     | |dark-horizontal|   |
+----------------+---------------------+
| Vertical       | |dark-vertical|     |
+----------------+---------------------+
| Logo Only      | |dark-notext|       |
+----------------+---------------------+

单色 (Light)
^^^^^^^^^^^^^^^^^^

.. note::

   浅色商标标记在透明上为浅灰色，因此在此页面上很难看到。它们想要使用在深色背景上。


+----------------+---------------------+
| Horizontal     | |light-horizontal|  |
+----------------+---------------------+
| Vertical       | |light-vertical|    |
+----------------+---------------------+
| Logo Only      | |light-notext|      |
+----------------+---------------------+

.. |color-horizontal| image:: images/statsmodels-logo-v2-horizontal.svg
   :width: 50%

.. |color-vertical| image:: images/statsmodels-logo-v2.svg
   :width: 14%

.. |color-notext| image:: images/statsmodels-logo-v2-no-text.svg
   :width: 9%

.. |dark-horizontal| image:: images/statsmodels-logo-v2-horizontal-dark.svg
   :width: 50%

.. |dark-vertical| image:: images/statsmodels-logo-v2-dark.svg
   :width: 14%

.. |dark-notext| image:: images/statsmodels-logo-v2-no-text-dark.svg
   :width: 9%

.. |light-horizontal| image:: images/statsmodels-logo-v2-horizontal-light.svg
   :width: 50%

.. |light-vertical| image:: images/statsmodels-logo-v2-light.svg
   :width: 14%

.. |light-notext| image:: images/statsmodels-logo-v2-no-text-light.svg
   :width: 9%
