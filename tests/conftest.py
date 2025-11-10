import warnings


def pytest_configure(config):
    config.addinivalue_line("markers", "optimization: long-running optimization tests")
    warnings.filterwarnings(
        "ignore",
        message=r"Complex dtype support in Diffrax.*",
        category=UserWarning,
        module=r"^equinox\._jit$",
    )
