import jax

jax.config.update("jax_enable_x64", True)
if jax.config.jax_platforms is None:
    jax.config.update("jax_platforms", "gpu")

import rydopt.gates as gates  # noqa: E402
import rydopt.pulses as pulses  # noqa: E402
import rydopt.simulation as simulation  # noqa: E402
import rydopt.optimization as optimization  # noqa: E402
import rydopt.characterization as characterization  # noqa: E402

__all__ = [
    "gates",
    "pulses",
    "simulation",
    "optimization",
    "characterization",
]
