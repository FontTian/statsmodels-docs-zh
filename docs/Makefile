# 最简单的生成 Sphinx 文档
#

# 通过代码命令可以设置如下变量.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = statsmodels
SOURCEDIR     = source
BUILDDIR      = build

PAPER         =
TOOLSPATH	  = ../tools/
DATASETBUILD  = dataset_rst.py
NOTEBOOKBUILD = nbgenerate.py
FOLDTOC  = fold_toc.py

# 内部变量.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS)


# 首先，使用不带参数的 make，如：make help：

	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)

.PHONY: help Makefile

cleanall: 
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
	-rm source/examples/generated/*
	-rm -rf source/examples/notebooks/generated/*
	-rm -rf ../tools/hash_dict.pickle
	-rm -rf source/datasets/generated/*

notebooks:
	@echo "Generating notebooks from examples/notebooks folder"
	$(TOOLSPATH)$(NOTEBOOKBUILD) --execute=True --allow_errors=True

html:
    # 制作图形目录
	@echo "Make static directory for images"
	mkdir -p $(BUILDDIR)/html/_static
	# generate the examples rst files
	@echo "Generating datasets from installed statsmodels.datasets"
	$(TOOLSPATH)$(DATASETBUILD)
	@echo "Executing notebooks from examples/notebooks folder"
	mkdir -p $(BUILDDIR)/source/examples/notebooks/generated
	$(TOOLSPATH)$(NOTEBOOKBUILD) --parallel --report-errors --skip-existing --execute-only
	@echo "Running sphinx-build"
	@echo @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)

# Catch-all target: 使用新的命令将所有未知的目标到 Sphinx
# "make mode" 选项.  $(O) 是r $(SPHINXOPTS)的快捷方式.
%: 生成文件
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
