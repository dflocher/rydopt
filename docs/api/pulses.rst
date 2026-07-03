rydopt.pulses
=============

.. currentmodule:: rydopt.pulses

.. autoclass:: PulseAnsatzFunction
   :members:

.. autoclass:: PulseAnsatz
   :members:

.. autoclass:: TwoPhotonPulseAnsatz
   :members:

.. autoclass:: PulseFamilyAnsatz
   :members:

.. autoclass:: PolynomialPulseMap
   :members:

.. autoclass:: PolynomialPulseMapWithCustomDuration
   :members:

.. autofunction:: empirical_cphase_duration

.. autoclass:: PulseParams

.. autoclass:: PulseFamilyParams

.. py:type:: ParamsFloatLike
   :canonical: PulseParams[float] | PulseFamilyParams[float] | Sequence[float] | jax.Array | numpy.ndarray | tuple[jax.Array, jax.Array, jax.Array, jax.Array]

   Pulse configuration as either
   ``PulseParams(duration, detuning_params, phase_params, rabi_params)``,
   ``PulseFamilyParams(duration_params, detuning_params, phase_params, rabi_params)``,
   an unpacked parameter tuple, or a packed parameter array/sequence.

   - **duration** / **duration_params** - Gate duration or pulse-family duration parameters
   - **detuning_params** - Parameters for the detuning sweep
   - **phase_params** - Parameters for the phase sweep
   - **rabi_params** - Parameters for the Rabi frequency amplitude sweep

.. py:type:: ParamsBoolLike
   :canonical: PulseParams[bool] | PulseFamilyParams[bool] | Sequence[bool] | jax.Array | numpy.ndarray

   Boolean masks as either
   ``PulseParams(fixed_duration, fixed_detuning_params, fixed_phase_params, fixed_rabi_params)``,
   ``PulseFamilyParams(fixed_duration_params, fixed_detuning_params, fixed_phase_params, fixed_rabi_params)``,
   or a packed boolean mask array/sequence, marking which pulse parameters are held constant during optimization.

   - **fixed_duration** / **fixed_duration_params** - Whether the duration or duration parameters are fixed
   - **fixed_detuning_params** - Boolean mask of fixed detuning parameters
   - **fixed_phase_params** - Boolean mask of fixed phase parameters
   - **fixed_rabi_params** - Boolean mask of fixed Rabi frequency amplitude parameters

General Pulse Ansatz Functions
------------------------------

.. automodule:: rydopt.pulses.general_pulse_ansatz_functions
   :members:

Soft-Box Pulse Ansatz Functions
-------------------------------

.. automodule:: rydopt.pulses.softbox_pulse_ansatz_functions
   :members:
