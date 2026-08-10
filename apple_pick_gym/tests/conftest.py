"""Pytest hooks for apple_pick_gym tests."""


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: long-horizon stability or optional benchmark-style tests",
    )
