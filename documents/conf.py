# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
from pathlib import Path
import sys

import sphinx_rtd_theme
from sphinx.application import Sphinx
from sphinx.ext.apidoc import main as sphinx_apidoc

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "exogibbs"
API_REFERENCE_ROOT = Path(__file__).resolve().parent / "exogibbs"
sys.path.insert(0, str(SOURCE_ROOT))

import exogibbs

expected_exogibbs_root = (SOURCE_ROOT / "exogibbs").resolve()
imported_exogibbs_root = Path(exogibbs.__file__).resolve().parent
if imported_exogibbs_root != expected_exogibbs_root:
    raise RuntimeError(
        "Imported exogibbs from outside this repository: "
        f"{imported_exogibbs_root} != {expected_exogibbs_root}"
    )


def _generate_api_reference(_app: Sphinx) -> None:
    """Generate a fresh API reference before Sphinx discovers source files."""
    sphinx_apidoc(
        [
            "--force",
            "--remove-old",
            "--tocfile",
            "index",
            "--output-dir",
            str(API_REFERENCE_ROOT),
            str(PACKAGE_ROOT),
            str(PACKAGE_ROOT / "exogibbs_version.py"),
        ]
    )


def setup(app: Sphinx) -> None:
    """Register documentation build hooks."""
    app.connect("builder-inited", _generate_api_reference)


# -- Project information -----------------------------------------------------

project = "ExoGibbs"
copyright = "2025, ExoGibbs contributors"
author = "ExoGibbs contributors"

# The short and full versions, including alpha/beta/rc tags.
version = "0.5"
release = "0.5.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/exogibbs_logo.png"

# html_theme_options = {
#    'style_nav_header_background': '#333',
# }
# Sphinx-Gallery
from sphinx_gallery.sorting import FileNameSortKey
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples"],
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": "/plot_",
    "ignore_pattern": "/_",
    "backreferences_dir": "backreferences",
    "doc_module": ("exogibbs",),
    "reference_url": {"exogibbs": None},
}
