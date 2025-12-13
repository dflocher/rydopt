.. _gates:

rydopt.gates
============

A gate systems class specifies (i) the physical system for implementing a gate and (ii) the target gate which should be
implemented. The class implements all methods from the :class:`GateSystem <rydopt.protocols.GateSystem>` protocol as defined in the Reference of
Internal Functions. This allows RydOpt's optimizer to calculate the time evolution of the physical system for a
given :class:`PulseAnsatz <rydopt.pulses.PulseAnsatz>` and to adapt the pulse parameters so that the infidelity with respect to the target gate is
minimized.

Rydberg Gate Systems
--------------------

The Rydberg gate systems describe gates that make use of the Rydberg interaction. They implement in addition to the methods from
the :class:`GateSystem <rydopt.protocols.GateSystem>` protocol, the methods from the :class:`RydbergSystem <rydopt.protocols.RydbergSystem>`
protocol. This allows to determine the time spent in the Rydberg state.

The different classes defer by the number of atoms and the conceptual atomic arrangement. An object can be constructed by
specifying:

1. the specific physical setting, i.e., the Rydberg-interaction strengths between the atoms, and the Rydberg-state decay rate.

2. the specific target gate angles.

.. autoclass:: rydopt.gates.TwoQubitGate
   :no-members:

.. autoclass:: rydopt.gates.ThreeQubitGateIsosceles
   :no-members:

.. autoclass:: rydopt.gates.FourQubitGatePyramidal
   :no-members:
