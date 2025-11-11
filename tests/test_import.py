"""Minimal smoke tests for the cellarc package."""


def test_version_exposed() -> None:
    import cellarc

    assert isinstance(cellarc.__version__, str)
    assert cellarc.__version__
