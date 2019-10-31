:orphan:

.. _statsmodels-examples:

例子
========

该页面提供了一系列示例、教程和食谱，用来帮助您入门 ``statsmodels``. 此处展示的每个示例
都可以作为 IPython Notebook 和纯 python 脚本用在 `statsmodels github
repository <https://github.com/statsmodels/statsmodels/tree/master/examples>`_.

我们还鼓励用户在 `Examples wiki page <https://github.com/statsmodels/statsmodels/wiki/Examples>`_ 
提交自己的示例、教程或很酷的statsmodels技巧

{# 此内容对空格敏感。不要重新格式化 #}

{% for category in examples%}
{% set underscore = "-" * (category.header | length) %}
{{ category.header }}
{{ underscore }}

.. toctree::
   :maxdepth: 1
   :hidden:

{% for notebook in category.links  %}   {{ notebook.target | replace('.html','') }}
{% endfor %}

{%- for notebook in category.links  %}
{% set heading = "`" ~ notebook.text ~ " <" ~ notebook.target|e ~ ">`_" %}
{% set subunderscore = "~" * (heading | length) %}
{{ heading }}
{{ subunderscore }}
.. image:: {{ notebook.img }}
   :target: {{ notebook.target }}
   :width: 360px

{%- endfor %}

{% endfor %}
