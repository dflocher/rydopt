rydopt.pulses
=============

.. currentmodule:: rydopt.pulses

Pulse Ansatz Functions
----------------------

A pulse ansatz function is a function of time that additionally takes a set of parameters as input. It describes the time evolution of, e.g., the
laser phase or the laser detuning. Below, we list pre-implemented pulse ansatz functions that can be used right away.

.. autoclass:: PulseAnsatzFunction
   :members:

General Pulse Ansatz Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. container:: toggle, toggle-hidden

    .. automodule:: rydopt.pulses.general_pulse_ansatz_functions
        :members:

Soft-Box Pulse Ansatz Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. container:: toggle, toggle-hidden

    .. automodule:: rydopt.pulses.softbox_pulse_ansatz_functions
        :members:


.. currentmodule:: rydopt.pulses

.. _pulse_ansatz_classes:

Pulse Ansatz Classes
--------------------

An object of a pulse ansatz class is used to describe a complete laser pulse. Here, we provide two classes: one that describes a single-photon pulse
and one that describes a two-photon pulse. Both classes implement the protocol :class:`PulseAnsatz <rydopt.protocols.PulseAnsatz>`.

.. autoclass:: SinglePhotonPulseAnsatz
   :members:

.. autoclass:: TwoPhotonPulseAnsatz
   :members:

Pulse Family Ansatz
-------------------

.. autoclass:: PulseFamilyAnsatz
   :members:

Pulse Maps
~~~~~~~~~~
.. container:: toggle, toggle-hidden

    .. autoclass:: PolynomialPulseMap
       :members:

    .. autoclass:: PolynomialPulseMapWithCustomDuration
       :members:

    .. autofunction:: empirical_cphase_duration

Pulse Parameters
----------------

The parameters of a pulse can be specified in several ways. We provide classes that can be used to specify all parameters of a pulse or a
pulse family. Moreover, we provide types that indicate how parameters may be specified without using the respective class.

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
