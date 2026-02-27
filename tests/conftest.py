import os
import warnings

from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "optimization: long-running optimization tests")

    warnings.filterwarnings(
        "ignore",
        message=r"Complex dtype support in Diffrax.*",
        category=UserWarning,
        module=r"^equinox\._jit$",
    )

    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"  # "cuda,cpu"
