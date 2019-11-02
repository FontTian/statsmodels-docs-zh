|Travis Build Status| |Azure CI Build Status| |Appveyor Build Status| |Coveralls Coverage| |Statsmodels中文文档|


为什么翻译
=================
本人会计出身，转行数据分析（更多是用excel、sql、spss等工具），30岁开始入坑python，在学习statsmodels包的过程中，在知乎、百度、CSDN和github上均没有找到statsmodels的中文文档，一开始以为 AphcheCN 开源组织（号称一群有想法，爱装逼，爱斗图，有活力，爱搞事，爱吃辣条的朋友）已经翻译了该包的文档，最后发现他们只是翻译该文档的目录，为了更好的入坑，我只好用我那拙劣的英语、再加上谷歌翻译，苦逼的翻译2周，也仅仅是把docs文件夹下的文件翻译完（未校验），水平有限，各位请多担待。翻译这个活没有什么技术可言，但真的很苦逼。所以真的非常的感谢 AphcheCN 开源组织在我学习的过程中贡献的各种中文文档，非常感谢！

关于统计模型
=================

statsmodels是一个Python软件包，为scipy提供了补充，以进行统计计算，包括描述性统计以及统计模型的估计和推断。

statsmodels主要包括如下子模块：

回归模型：线性回归，广义线性模型，稳健的线性模型，线性混合效应模型等等。

方差分析（ANOVA）。

时间序列分析：AR，ARMA，ARIMA，VAR和其它模型。

非参数方法： 核密度估计，核回归。

统计模型结果可视化。

比较：statsmodels更关注统计推断，提供不确定估计和参数p-value。相反的，scikit-learn注重预测。


文献资料
=============

最新版本的文档位于

https://www.statsmodels.org/stable/

开发版本的文档位于

https://www.statsmodels.org/dev/

最新的改进在发行说明中突出显示

https://www.statsmodels.org/stable/release/version0.9.html

可从https://statsmodels.github.io/stable/ 和https://statsmodels.github.io/dev/获得文档的备份。



主要特点
=============

* 线性回归模型：

  - 普通最小二乘法
  - 广义最小二乘法
  - 加权最小二乘法
  - 具有自回归误差的最小二乘
  - 分位数回归
  - 递归最小二乘法

* 具有混合效应和方差成分的混合线性模型
* GLM：支持所有单参数指数族分布的广义线性模型
* 用于二项式和泊松的贝叶斯混合GLM
* GEE：单向聚类或纵向数据的广义估计方程
* 离散模型:

  - Logit 和 Probit
  - 多项 logit (MNLogit)
  - 泊松和广义泊松回归
  - 负二项式回归
  - 零膨胀计数模型
  
* RLM: 鲁棒的线性模型，支持多个 M 估计器。
* 时间序列分析：时间序列分析模型

  - 完整的StateSpace建模框架
  
    - 季节性ARIMA和ARIMAX模型
    - VARMA和VARMAX模型
    - 动态因子模型
    - 未观测到的组件模型

  - 马尔可夫切换模型（MSAR），也称为隐马尔可夫模型（HMM）
  - 单变量时间序列分析：AR，ARIMA
  - 矢量自回归模型，VAR和结构VAR
  - 矢量误差修正模型，VECM
  - 指数平滑，Holt-Winters
  - 时间序列的假设检验：单位根，协整和其他
  - 用于时间序列分析的描述性统计数据和过程模型
  
* 生存分析:

  - 比例风险回归（Cox模型）
  - 生存者函数估计（Kaplan-Meier）
  - 累积发生率函数估计

* 多变量:

  - 缺失数据的主成分分析
  - 旋转因子分析
  - MANOVA
  - 典型相关

* 非参数统计：单变量和多变量核密度估计
* 数据集：用于示例和测试的数据集
* 统计：广泛的统计检验

  - 诊断和规格检验
  - 拟合优度和正态性检验
  - 多元测试函数
  - 各种其他统计检验
  
* 使用MICE进行插补，秩序统计回归和高斯插补
* 调解分析
* 图形包括用于数据和模型结果的可视化分析的绘图功能


* 输入/输出

  - 用于读取Stata .dta文件的工具，但pandas具有较新的版本
  -  表输出到ascii，latex和html
  
* 其他模型

* Sandbox：statsmodels包含一个 sandbox 文件夹，其中包含处于开发和测试各个阶段的代码，
  因此不被视为“生产就绪”。其中包括：

  - 广义矩法（GMM）估计器
  - 核回归
  - scipy.stats.distributions的各种扩展
  - 面板数据模型
  - 信息理论测度

如何获得
=============
GitHub上的master分支是最新的代码

https://www.github.com/statsmodels/statsmodels

发行标签的源代码下载可在GitHub上获得

https://github.com/statsmodels/statsmodels/tags

二进制文件和源代码发行版可从PyPi获得

https://pypi.org/project/statsmodels/

二进制文件可以安装在Anaconda中

conda install statsmodels


Installing from sources
=======================

有关要求，请参阅INSTALL.txt或请参阅文档

https://statsmodels.github.io/dev/install.html

执照
=======

Modified BSD (3-clause)

讨论和发展
==========================

Discussions take place on our mailing list.

https://groups.google.com/group/pystatsmodels

我们对有关可用性的反馈和改进建议非常感兴趣。

Bug 报告
===========

Bug 报告可以提交至问题跟踪器，网址为

https://github.com/statsmodels/statsmodels/issues

.. |Travis Build Status| image:: https://travis-ci.org/statsmodels/statsmodels.svg?branch=master
   :target: https://travis-ci.org/statsmodels/statsmodels
.. |Azure CI Build Status| image:: https://dev.azure.com/statsmodels/statsmodels-testing/_apis/build/status/statsmodels.statsmodels?branch=master
   :target: https://dev.azure.com/statsmodels/statsmodels-testing/_build/latest?definitionId=1&branch=master
.. |Appveyor Build Status| image:: https://ci.appveyor.com/api/projects/status/gx18sd2wc63mfcuc/branch/master?svg=true
   :target: https://ci.appveyor.com/project/josef-pkt/statsmodels/branch/master
.. |Coveralls Coverage| image:: https://coveralls.io/repos/github/statsmodels/statsmodels/badge.svg?branch=master
   :target: https://coveralls.io/github/statsmodels/statsmodels?branch=master
