"""Common fixtures for the hand-built neural-network project."""

import pytest

from prepare import prepare_data


@pytest.fixture(scope="session")
def prepared():
    return prepare_data()
