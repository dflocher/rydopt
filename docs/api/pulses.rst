rydopt.pulses
=============

.. currentmodule:: rydopt.pulses

.. autoclass:: PulseAnsatz

.. py:type:: PulseParams
   :canonical: tuple[float, ArrayLike, ArrayLike, ArrayLike]

   Pulse configuration ``(duration, detuning_params, phase_params, rabi_params)``.

   - **duration** - Gate duration
   - **detuning_params** - Parameters for the detuning sweep
   - **phase_params** - Parameters for the phase sweep
   - **rabi_params** - Parameters for the Rabi frequency amplitude sweep

.. py:type:: FixedPulseParams
   :canonical: tuple[bool, ArrayLike, ArrayLike, ArrayLike]

   Boolean masks ``(fixed_duration, fixed_detuning_params, fixed_phase_params, fixed_rabi_params)`` marking
   which pulse parameters are held constant during optimization.

   - **fixed_duration** - Whether the gate duration is fixed
   - **fixed_detuning_params** - Boolean mask of fixed detuning parameters
   - **fixed_phase_params** - Boolean mask of fixed phase parameters
   - **fixed_rabi_params** - Boolean mask of fixed Rabi frequency amplitude parameters

.. automodule:: rydopt.pulses
   :members:
   :exclude-members: PulseAnsatz, PulseParams, FixedPulseParams
