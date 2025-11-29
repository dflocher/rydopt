# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RydOpt"
copyright = "2025, David Locher, Sebastian Weber, Jakob Holschbach"  # noqa: A001
author = "David Locher, Sebastian Weber, Jakob Holschbach"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_tabs.tabs",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autosummary and autodoc--------------------------------------
autosummary_ignore_module_all = False

add_module_names = False
autodoc_class_signature = "mixed"
autodoc_typehints = "description"

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
    "HamiltonianFunction": "HamiltonianFunction",
}

# -- Options for jupyter notebooks -------------------------------------------
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None).split("/")[-1] %}

.. raw:: html

    <style>
      .nbinput .prompt,
      .nboutput .prompt {
        display: none;
      }
    </style>

    <div class="admonition note">
      This page was generated from the Jupyter notebook
      <a class="reference external" href="{{ docname|e }}">{{ docname|e }}</a>.
      Open in
      <a class="reference external" href="https://colab.research.google.com/github/dflocher/rydopt/blob/main/docs/examples/{{ docname|e }}">Google Colab</a>.
    </div>
"""  # noqa: E501

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]
