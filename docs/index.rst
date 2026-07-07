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
in neutral atom quantum computing platforms. The optimization methods support GPUs and multi-core CPUs, using an
efficient implementation based on JAX.

Install the software with pip (requires Python ≥ 3.10; for enabling GPU support and tips, see our
:doc:`extended installation instructions <install>`):

.. code-block:: bash

   pip install rydopt


Citing RydOpt
-------------
If you find this library useful for your research, please cite:

    D.F. Locher, J. Old, K. Brechtelsbauer, J. Holschbach, H.P. Büchler, S. Weber, M. Müller,
    *Multiqubit Rydberg Gates for Quantum Error Correction*, `PRX Quantum 7, 020354 (2026) <https://doi.org/10.1103/j8fm-24cf>`_

Contributors
------------
The following people have, so far, contributed to the development of RydOpt:

- `David Locher <https://github.com/dflocher>`_
- `Sebastian Weber <https://github.com/seweber>`_
- Jakob Holschbach
- `Javad Kazemi <https://github.com/jakazemi>`_

We warmly welcome new contributions! Please refer to the :doc:`contributor guide <contribute/development>` for more information!

The development of RydOpt has been supported by `Forschungszentrum Jülich <https://www.fz-juelich.de/>`_,
`RWTH Aachen University <https://www.rwth-aachen.de/>`_, `University of Stuttgart <https://www.uni-stuttgart.de/>`_, and the
company `ParityQC <https://parityqc.com/>`_. We acknowledge support from the Federal Ministry of Research, Technology and Space (BMFTR) through the
grant `MUNIQC-Atoms <https://muniqc-atoms.munich-quantum-valley.de/>`_ and from the German Research Foundation (DFG) through the priority
programme `SPP 2514 <https://www.spp2514.kit.edu/>`_.

.. image:: _static/MUNIQC_Atoms_Logo.svg
    :width: 300px
    :target: https://muniqc-atoms.munich-quantum-valley.de/
.. image:: _static/SPP_Logo.png
    :width: 130px
    :target: https://www.spp2514.kit.edu/
.. image:: _static/ParityQC_Logo.svg
    :width: 220px
    :target: https://parityqc.com/

License
-------
The RydOpt software is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   about
   install
   concepts
   tutorials

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api/gates
   api/pulses
   api/simulation
   api/optimization
   api/characterization

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contributor Guide

   contribute/development
   contribute/internal
