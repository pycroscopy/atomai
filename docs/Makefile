# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: gh-pages
.ONESHELL:
gh-pages:
   rm -rf /tmp/gh-pages
   cp -r $(BUILDDIR)/html /tmp/gh-pages
   git checkout gh-pages
   cd .. && rm -rf * && cp -r /tmp/gh-pages/* . && git add . && git commit -m "Updated gh-pages" && git push && git checkout master
