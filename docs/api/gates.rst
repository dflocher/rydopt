.. _gates:

rydopt.gates
============

A gate class implements a :ref:`Protocol <label-protocols>` as defined in the Reference of Internal Functions.

The Rydberg gate classes implemented here specify the number of atoms and the conceptual atomic arrangement.
A Rydberg gate object then fixes

(i) the specific physical setting, i.e., the Rydberg-interaction strengths between the atoms, and the Rydberg-state decay rate.

(ii) the specific target gate angles.


.. autoclass:: rydopt.gates.TwoQubitGate
   :no-members:

.. autoclass:: rydopt.gates.ThreeQubitGateIsosceles
   :no-members:

.. autoclass:: rydopt.gates.FourQubitGatePyramidal
   :no-members:
