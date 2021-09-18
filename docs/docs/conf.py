# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_theme
# import sphinx_material
# import murray
# import tibas.tt
# import alabaster
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
project = 'DDP'
author = 'TVS'

html_show_sourcelink = False


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.doctest",
  "sphinx.ext.extlinks",
  "sphinx.ext.intersphinx",
  'sphinx.ext.autosummary',
  'fluiddoc.mathmacro',
  "sphinx.ext.todo",
  "sphinx.ext.mathjax",
  "sphinx.ext.viewcode",
  "sphinxcontrib.bibtex"
]

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'super'
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'furo'
# html_theme = 'murray'
# html_theme = 'tt'
html_theme = 'sphinx_book_theme'
# html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
# html_theme_path = [murray.get_html_theme_path()]
# html_theme_path = [tibas.tt.get_path(), alabaster.get_path()]
html_show_copyright = False
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_sidebars = {
    # "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_title = 'DDP Wrapper'
html_theme_options = {
  # 'color_primary': 'teal',
  # 'color_accent': 'light-teal',
  'use_download_button': False,
  'use_fullscreen_button': False,
  'extra_navbar': '<span></span>',
  # "single_page": True
}

html_context = {
  "display_github": False,  # Add 'Edit on Github' link instead of 'View page source'
  "github_user": "sujaltv",
  "github_repo": "phd",
  "github_version": "master"
  # "source_suffix": source_suffix,
}