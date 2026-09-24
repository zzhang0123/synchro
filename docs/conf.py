"""Sphinx configuration for the synchro documentation (Read the Docs)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import synchro  # noqa: E402  (enables float64; provides the version)

project = "synchro"
author = "Zheng Zhang and Jens Chluba"
copyright = "2026, Zheng Zhang and Jens Chluba"
release = synchro.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# README sections are included into guide pages; their heading levels start
# below the page title, which MyST reports as skipped levels.
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3
suppress_warnings = ["myst.header"]

autodoc_default_options = {"members": True, "member-order": "bysource"}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = f"synchro {release}"
html_theme_options = {
    "source_repository": "https://github.com/zzhang0123/synchro/",
    "source_branch": "main",
    "source_directory": "docs/",
}
