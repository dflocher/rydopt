About
=====

Neutral Atom Quantum Computing
------------------------------

Neutral atoms trapped in optical tweezer arrays are among the most promising platforms for building a fault-tolerant quantum computer.
Two low-energy electronic states of an atom encode the computational basis states :math:`|0\rangle` and :math:`|1\rangle`.
The mechanism underlying entangling operations is strong interactions between atoms in highly excited Rydberg states.
These interactions can be turned on and off 'on demand', by exciting atoms to those Rydberg states :math:`|r\rangle` with a laser.
Two- and multiqubit gates are thus realized by laser pulses, which are variations in time of the laser detuning :math:`\Delta(t)`,
and the Rabi frequency :math:`|\Omega(t)|e^{i\xi(t)}`, where :math:`\xi(t)` is the laser phase and :math:`|\Omega(t)|` is the
Rabi frequency amplitude.
The dynamics of a set of :math:`N` atoms subjected to the very same laser pulse is described by the Hamiltonian

.. math::
    H(t) = \hbar \sum_{i=1}^{N} \frac{1}{2} \big( \Omega(t) |r_i \rangle \langle 1_i| + \mathrm{h.c.} \big)
    + \Delta(t) |r_i\rangle \langle r_i| + \sum_{i,j<i} V_{ij} | r_i r_j \rangle \langle r_i r_j | ,

where :math:`V_{ij}` describes the interaction between atom :math:`i` and atom :math:`j`.

The RydOpt package
------------------

This package provides tools to simulate the dynamics of multiple atoms subjected to a gate pulse.
In particular, an optimization tool can identify pulses that implement a desired target gate on a set of atoms.
