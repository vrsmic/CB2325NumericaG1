import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CB2325NumericaG1'
copyright = '2025, Alexander Kahleul, Cauan Carlos Rodrigues Dutra, Juan Martins Santos, Luana Fagundes De Lima, Luana Mognon Da Silva, Lucas Fraga Damasceno, Mariana Tiemi Yoshioka, Mateus Stacoviaki Galvão, Micaele Magalhães Brandão Veras, Rafael Augusto De Almeida, Ryan Carvalho Pereira Dos Santos'
author = 'Alexander Kahleul, Cauan Carlos Rodrigues Dutra, Juan Martins Santos, Luana Fagundes De Lima, Luana Mognon Da Silva, Lucas Fraga Damasceno, Mariana Tiemi Yoshioka, Mateus Stacoviaki Galvão, Micaele Magalhães Brandão Veras, Rafael Augusto De Almeida, Ryan Carvalho Pereira Dos Santos'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon', 
]

templates_path = ['_templates']
exclude_patterns = []

language = 'pt_BR'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
