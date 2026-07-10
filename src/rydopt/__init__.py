import os

if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if "JAX_ENABLE_X64" not in os.environ:
    os.environ["JAX_ENABLE_X64"] = "true"

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax

# Cache compiled XLA programs on disk so repeated runs (e.g., new processes optimizing
# the same gate/pulse combination) skip the expensive JIT compilation.
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    _cache_home = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    jax.config.update("jax_compilation_cache_dir", os.path.join(_cache_home, "rydopt", "jax_cache"))
if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import rydopt.characterization as characterization
import rydopt.gates as gates
import rydopt.optimization as optimization
import rydopt.pulses as pulses
import rydopt.simulation as simulation

warnings.filterwarnings(
    "ignore",
    message=r"Complex dtype support in Diffrax.*",
    category=UserWarning,
    module=r"^equinox\._jit$",
)

__all__ = [
    "gates",
    "pulses",
    "simulation",
    "optimization",
    "characterization",
]
