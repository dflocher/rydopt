RydOpt - A Multiqubit Rydberg Gate Optimizer
============================================

.. image:: https://readthedocs.org/projects/rydopt/badge/?version=latest
   :target: http://rydopt.readthedocs.io
   :alt: docs

.. image:: https://github.com/dflocher/rydopt/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/dflocher/rydopt/actions/workflows/tests.yml
   :alt: tests

.. image:: https://img.shields.io/pypi/v/rydopt.svg?style=flat
   :target: https://pypi.org/project/rydopt/
   :alt: pypi

RydOpt is a Python package for the optimization of laser pulses implementing two- and multiqubit Rydberg gates
in neutral atom quantum computing platforms. The opimization methods support GPUs and multi-core CPUs, using an
efficient implementation based on JAX.

Install the software with pip (requires Python ≥ 3.10, for enabling GPU support and tips see our
:doc:`extended installation instructions <install>`):

.. code-block:: bash

   pip install rydopt

If you find this library useful for your research, please cite:

    David F. Locher, Josias Old, Katharina Brechtelsbauer, Jakob Holschbach, Hans Peter Büchler, Sebastian Weber, and Markus Müller,
    *Multiqubit Rydberg Gates for Quantum Error Correction* (publication pending)

The RydOpt software is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   about
   install
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/gates
   api/pulses
   api/simulation
   api/optimization
   api/characterization

.. toctree::
   :maxdepth: 1
   :caption: Contributor Guide

   contribute/development
   contribute/internal
