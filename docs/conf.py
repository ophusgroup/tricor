# Configuration file for the Sphinx documentation builder.

project = "tricor"
copyright = "2025, Colin Ophus"
author = "Colin Ophus"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]

# MyST (Markdown) support
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "dollarmath",
    "amsmath",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Theme
html_theme = "furo"
html_title = "tricor"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2a6e4e",
        "color-brand-content": "#2a6e4e",
    },
}

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# Napoleon (Google/NumPy docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# myst-nb: execute code cells during build
nb_execution_mode = "auto"  # execute cells marked with {code-cell}
nb_execution_timeout = 120  # seconds per cell
nb_execution_raise_on_error = True
