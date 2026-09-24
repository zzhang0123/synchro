"""Shared fixtures for ``tests/model``: small truncations and moment indices.

These fixtures need no ``Channels`` or ``Reference``; implementers of the
other modules may add their own fixtures below without changing these.
"""

import pytest

from synchro.model.index import MomentIndex, Truncation


@pytest.fixture(scope="session")
def truncation_111():
    """``(L_mu, L_eta, N) = (1, 1, 1)``: ``n0=12, n2=8, n_real=28``."""
    return Truncation(1, 1, 1)


@pytest.fixture(scope="session")
def truncation_222():
    """``(L_mu, L_eta, N) = (2, 2, 2)``: ``n0=54, n2=50, n_real=154``."""
    return Truncation(2, 2, 2)


@pytest.fixture(scope="session")
def index_111(truncation_111):
    return MomentIndex.build(truncation_111)


@pytest.fixture(scope="session")
def index_222(truncation_222):
    return MomentIndex.build(truncation_222)
