import os

if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if "JAX_ENABLE_X64" not in os.environ:
    os.environ["JAX_ENABLE_X64"] = "true"

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import rydopt.characterization as characterization
import rydopt.gates as gates
import rydopt.optimization as optimization
import rydopt.pulses as pulses
import rydopt.simulation as simulation

__all__ = [
    "characterization",
    "gates",
    "optimization",
    "pulses",
    "simulation",
]
