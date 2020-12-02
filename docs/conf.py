# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("sphinx_ext"))

from wildwood import *

from github_link import make_linkcode_resolve

# -- Project information -----------------------------------------------------

project = "wildwood"
copyright = "2020, Stéphane Gaïffas"
author = "Stéphane Gaïffas"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    #    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
autodoc_default_options = {"members": None, "inherited-members": None}


# sphinx_gallery_conf = {
#     "examples_dirs": "../examples",
#     "doc_module": "linlearn",
#     "gallery_dirs": "auto_examples",
#     "ignore_pattern": "../run_*|../playground_*",
#     "backreferences_dir": os.path.join("modules", "generated"),
#     "show_memory": False,
#     "reference_url": {"onelearn": None},
# }


linkcode_resolve = make_linkcode_resolve(
    "wildwood",
    u"https://github.com/pyensemble/"
    "wildwood/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"

# Generate the plots for the gallery
plot_gallery = "True"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

html_sidebars = {
    "**": ["about.html", "navigation.html", "searchbox.html"],
    "auto_examples": ["index.html"],
}
html_theme_options = {
    "description": "scikit-learn compatible alternative random forests algorithms",
    "github_user": "pyensemble",
    "github_repo": "wildwood",
    "github_button": True,
    "fixed_sidebar": True,
    "travis_button": False,
    "logo_text_align": "center",
    "github_banner": True,
}

html_logo = "images/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

from datetime import datetime

now = datetime.now()
html_show_copyright = copyright = (
    str(now.year)
    + ', <a href="https://github.com/linlearn/linlearn/graphs/contributors">linlearn developers</a>. Updated on '
    + now.strftime("%B %d, %Y")
)


# intersphinx_mapping = {
#     "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
#     "numpy": ("https://docs.scipy.org/doc/numpy/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
#     "matplotlib": ("https://matplotlib.org/", None),
#     "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
#     "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
#     "sklearn": ("https://scikit-learn.org/stable/", None),
# }
