# 文档的文档

我们的文档使用了一系列的 sphinx 文档和jupyter notebooks。 Jupyter notebooks 
应用于展示一个更长的、独立的示例。Sphinx文档也很不错，对于目录表和API文档更加合适。

## 建立过程

构建文档需要一些其他的依赖关系。您可以通过以下方式获得大多数

```bash

   pip install -e .[docs]

```

从项目的根开始，一些示例依赖于 rpy2 来执行R代码文档，由于难以安装，因此它并不包含在安装要求中。

要生成 HTML 文档, 从docs目录中运行make html，这将执行一下类型的文件

1. datasets
2. notebooks
3. sphinx

# Notebook Builds

我们用 nbconvert 来执行 notebook，然后将它们转换为HTML。转由statsmodels/tools/nbgenerate.py 处理。默认的python内核（嵌入在笔记本中）为python3。您至少nbconvert==4.2.0需要指定一个非默认内核，该内核可以在Makefile中传递。
