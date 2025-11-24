# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rydopt"
copyright = "2025, rydopt developers"
author = "rydopt developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autosummary and autodoc--------------------------------------
autosummary_ignore_module_all = False

add_module_names = False
autodoc_class_signature = "mixed"
autodoc_typehints = "signature"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": True,
    "class-doc-from": "class",
}

autodoc_type_aliases = {
    "ParamsTuple": "ParamsTuple",
    "FixedParamsTuple": "FixedParamsTuple",
    "PulseAnsatzFunction": "PulseAnsatzFunction",
    "PulseFunction": "PulseFunction",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_logo = "_static/logo.png"
html_static_path = ["_static"]
