.. _gates:

rydopt.gates
============

A class representing a gate system specifies (i) the physical system for implementing a gate and (ii) the target gate unitary which should be
executed. The class implements all methods from the :class:`GateSystem <rydopt.protocols.GateSystem>` protocol as defined in the Reference of
Internal Functions. This allows RydOpt's optimizer to calculate the time evolution of the physical system for a
given :class:`PulseAnsatz <rydopt.pulses.PulseAnsatz>` and to adapt the pulse parameters so that the infidelity with respect to the target gate is
minimized.

Rydberg Gate Systems
--------------------

Rydberg gate systems describe gates that make use of the Rydberg interaction. In addition to the methods from
the :class:`GateSystem <rydopt.protocols.GateSystem>` protocol, they implement the methods from the :class:`RydbergSystem <rydopt.protocols.RydbergSystem>`
protocol. This allows one to determine the time spent in Rydberg states.

The different classes below differ by the number of atoms and the conceptual atomic arrangement.
An object is constructed by specifying:

1. the specific physical setting, i.e., the Rydberg-interaction strengths between the atoms, and the Rydberg-state decay rate.

2. the specific target gate angles.

Symmetric
~~~~~~~~~

The following classes implement gate systems corresponding to specific symmetric atom arrangements. Each atoms is driven with the same Rabi frequency.

.. autoclass:: rydopt.gates.TwoQubitGate
   :no-members:

.. autoclass:: rydopt.gates.ThreeQubitGateIsosceles
   :no-members:

.. autoclass:: rydopt.gates.FourQubitGatePyramidal
   :no-members:

Asymmetric
~~~~~~~~~~

The following classes work for arbitrary atom arrangements. Optionally, each atom can be driven with an individually scaled
Rabi frequency, specified by the per-atom scaling factors ``s1``, ``s2``, … .

.. autoclass:: rydopt.gates.TwoQubitGateAsym
   :no-members:

.. autoclass:: rydopt.gates.ThreeQubitGateAsym
   :no-members:

.. autoclass:: rydopt.gates.FourQubitGateAsym
   :no-members:
