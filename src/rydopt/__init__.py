import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_enable_x64", True)
if jax.config.jax_platforms is None:
    jax.config.update("jax_platforms", "cpu")

import rydopt.characterization as characterization  # noqa: E402
import rydopt.gates as gates  # noqa: E402
import rydopt.optimization as optimization  # noqa: E402
import rydopt.pulses as pulses  # noqa: E402
import rydopt.simulation as simulation  # noqa: E402

__all__ = [
    "characterization",
    "gates",
    "optimization",
    "pulses",
    "simulation",
]
