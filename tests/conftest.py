import os


def pytest_configure(config):
    config.addinivalue_line("markers", "optimization: long-running optimization tests")
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"  # "cuda,cpu"
