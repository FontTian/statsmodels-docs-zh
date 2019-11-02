.. module:: statsmodels.base.distributed_estimation
.. currentmodule:: statsmodels.base.distributed_estimation

处理大数据
============================

大数据是现代世界中的流行语。尽管 statsmodels 可以将加载到内存中的中小型数据集（可能数以万计的观测值）进行使用，
但存在有数百万个观察值或更多观测值的时候，根据你的事例，statsmodels 可能是合适的工具，也可能不是。

statsmodels 及其编写的大多数软件堆栈都在内存中运行。结果，在更大的数据集上建立模型可能具有挑战性甚至不切实际。
话虽如此，但也有两种使用 statsmodels 处理在较大数据集上构建模型的通用策略。

Divide and Conquer - Distributing Jobs 分而治之- 分布式工作
--------------------------------------

如果您的系统能够加载所有数据，但是您尝试执行的分析速度很慢，则可以在数据的水平切片上构建模型，然后在适合时汇总各个模型。

这种方法的当前局限性是，statsmodels 通常不支持在 `patsy <https://patsy.readthedocs.io/en/latest/>`_ 中来构建你的设计矩阵 (被称为 `exog`) ，这是很有挑战的
更多详细示例 `here <examples/notebooks/generated/distributed_estimation.html>`_.

.. autosummary::
   :toctree: generated/

   DistributedModel
   DistributedResults

数据子集
--------------------

如果整个数据集太大而无法存储在内存中，则可以尝试将其存储在 `Apache Parquet <https://parquet.apache.org/>`_
or `bcolz <http://bcolz.blosc.org/en/latest/>`_. 的列容器中。通过使用 patsy 公式接口，将可以使用 `__getitem__` 函数 (即 data['Item']) 提取指定的列。

.. code-block:: python

    import pyarrow as pa
    import pyarrow.parquet as pq
    import statsmodels.formula.api as smf

    class DataSet(dict):
        def __init__(self, path):
            self.parquet = pq.ParquetFile(path)

        def __getitem__(self, key):
            try:
                return self.parquet.read([key]).to_pandas()[key]
            except:
                raise KeyError

    LargeData = DataSet('LargeData.parquet')

    res = smf.ols('Profit ~ Sugar + Power + Women', data=LargeData).fit()

此外，您可以将代码添加到此示例 `DataSet` 对象中，仅返回部分行，直到建立了一个好的模型为止。
然后，您可以根据更多数据调整最终模型。
