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
    "PulseParams": "PulseParams",
    "FixedPulseParams": "FixedPulseParams",
    "ArrayLike": "ArrayLike",
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

      .rst-content p {
        margin-bottom: 15px;
      }

      .rst-content div.nbinput.nblast.docutils.container + p,
      .rst-content div.nboutput.nblast.docutils.container + p {
        margin-top: 15px;
      }
    </style>

    <div class="admonition note">
      This page was generated from the Jupyter notebook
      <a class="reference external" href="{{ docname|e }}">{{ docname|e }}</a>.
      Open in
      <a class="reference external" href="https://colab.research.google.com/github/dflocher/rydopt/blob/main/docs/examples/{{ docname|e }}">Google Colab</a>.
    </div>
"""  # noqa: E501

pygments_style = "friendly"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]

# -- Work around Sphinx not linking `py:type` in type annotations ------------
# When a type annotation produces a py:class reference to one of our
# aliases and Sphinx can't resolve it, redirect the lookup to `py:type`.


def resolve_type_aliases(app, env, node, contnode):
    if node.get("refdomain") != "py":
        return None

    # Annotations usually generate a "class" reference (even for aliases)
    if node.get("reftype") != "class":
        return None

    target = node.get("reftarget")
    if target not in {"PulseParams", "FixedPulseParams"}:
        return None

    # Try to resolve as a py:type instead
    py_domain = app.env.get_domain("py")
    return py_domain.resolve_xref(env, node["refdoc"], app.builder, "type", target, node, contnode)


def setup(app):
    app.connect("missing-reference", resolve_type_aliases)
