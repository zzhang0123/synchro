"""Index bookkeeping against an independent NumPy/itertools enumeration.

The oracle enumerates every ``(l, k, r, s, b)`` row with ``itertools`` and
counts with ``math.comb``; it shares no code with ``synchro.model.index``.
Pinned numbers come from FINAL_DESIGN.md Section 3 and INTERFACES.md.
"""

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synchro.model.index import Entry, MomentIndex, Truncation

# ----------------------------------------------------------------------
# Independent oracle
# ----------------------------------------------------------------------


def oracle_rows(L_mu, L_eta, N, depth_degree=None, *, parity=True, with_V=True):
    """Return (h0, h0_ext, h2) as sorted lists of 5-tuples."""
    pairs = list(itertools.product(range(L_mu + 1), range(L_eta + 1)))
    h0 = [
        (l, k, r, s, 0)
        for (l, k) in pairs
        if with_V or (l + k) % 2 == 0
        for r in range(N + 1)
        for s in range(N + 1)
        if r + s <= N
    ]
    p_pairs = [(l, k) for (l, k) in pairs if (not parity) or (l + k) % 2 == 0]
    max_b = N if depth_degree is None else depth_degree
    if depth_degree is None:
        rsb = [
            (r, s, b)
            for r in range(N + 1)
            for s in range(N + 1)
            for b in range(N + 1)
            if r + s + b <= N
        ]
    else:
        rsb = [
            (r, s, b)
            for r in range(N + 1)
            for s in range(N + 1)
            for b in range(max_b + 1)
            if r + s <= N
        ]
    h2 = [(l, k, r, s, b) for (l, k) in p_pairs for (r, s, b) in rsb]
    h0_ext = [row for row in h2 if row[4] >= 1]
    return sorted(h0), sorted(h0_ext), sorted(h2)


def oracle_sizes(L_mu, L_eta, N, depth_degree=None):
    n0 = (L_mu + 1) * (L_eta + 1) * math.comb(N + 2, 2)
    d = 1 if (L_mu % 2 == 0 and L_eta % 2 == 0) else 0
    n_plus = ((L_mu + 1) * (L_eta + 1) + d) // 2
    if depth_degree is None:
        n2 = n_plus * math.comb(N + 3, 3)
    else:
        n2 = n_plus * math.comb(N + 2, 2) * (depth_degree + 1)
    return n0, n2, n0 + 2 * n2


# ----------------------------------------------------------------------
# Truncation
# ----------------------------------------------------------------------


def test_truncation_max_b_rule():
    assert Truncation(2, 2, 2).max_b() == 2
    assert Truncation(2, 2, 2, depth_degree=3).max_b() == 3
    assert Truncation(2, 2, 2, depth_degree=0).max_b() == 0
    assert Truncation(0, 0, 0).max_b() == 0


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1.0, 1, 1),
        (True, 1, 1),
        (1, 1, 1, -1),
        (1, 1, 1, 2.0),
        (1, 1, 1, False),
    ],
)
def test_truncation_rejects_bad_ints(args):
    with pytest.raises(ValueError):
        Truncation(*args)


def test_truncation_is_hashable_and_static():
    t = Truncation(2, 2, 2)
    assert hash(t) == hash(Truncation(2, 2, 2))
    assert t == Truncation(2, 2, 2)
    assert t != Truncation(2, 2, 2, depth_degree=2)
    assert jax.tree_util.tree_leaves(t) == []


# ----------------------------------------------------------------------
# Pinned worked numbers (FINAL_DESIGN.md Section 3, INTERFACES.md)
# ----------------------------------------------------------------------


def test_sizes_222(index_222):
    assert (index_222.n0, index_222.n2, index_222.n_real) == (54, 50, 154)
    assert index_222.n_lk == 9
    assert len(index_222.h0_ext) == 20


def test_sizes_111(index_111):
    assert (index_111.n0, index_111.n2, index_111.n_real) == (12, 8, 28)
    assert index_111.n_lk == 4


def test_sizes_002():
    idx = MomentIndex.build(Truncation(0, 0, 2))
    assert (idx.n0, idx.n2, idx.n_real) == (6, 10, 26)
    assert idx.pairs == ((0, 0),)


def test_sizes_appendix_c_layout():
    idx = MomentIndex.build(Truncation(2, 2, 2, depth_degree=3))
    assert (idx.n0, idx.n2, idx.n_real) == (54, 120, 294)
    assert idx.truncation.max_b() == 3
    assert max(row[4] for row in idx.h2) == 3
    # r+s <= N holds even with b = 3
    assert all(row[2] + row[3] <= 2 for row in idx.h2)


def test_pinned_slots_222(index_222):
    assert index_222.position(0, 0, 0, 0, 0, 0) == 0
    assert index_222.h0[0] == (0, 0, 0, 0, 0)
    assert index_222.position(0, 1, 2, 1, 0, 0) == 33
    assert index_222.position(2, 0, 2, 0, 1, 0) == 67
    assert index_222.position(2, 0, 2, 0, 1, 0) + index_222.n2 == 117


def test_h0_slot_formula_222(index_222):
    i_rs = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (2, 0): 5}
    for l in range(3):
        for k in range(3):
            for (r, s), i in i_rs.items():
                assert index_222.position(0, l, k, r, s, 0) == 6 * (3 * l + k) + i


def test_h2_slot_formula_222(index_222):
    p_lk = {(0, 0): 0, (0, 2): 1, (1, 1): 2, (2, 0): 3, (2, 2): 4}
    j_rsb = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 2, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0),
    ]
    for (l, k), p in p_lk.items():
        for j, (r, s, b) in enumerate(j_rsb):
            assert index_222.position(2, l, k, r, s, b) == 54 + 10 * p + j


def test_labels_222(index_222):
    labels = index_222.labels()
    assert len(labels) == 154
    assert labels[33] == "M0[l=1,k=2;r=1,s=0]"
    assert labels[0] == "M0[l=0,k=0;r=0,s=0]"
    slot = index_222.position(2, 0, 2, 0, 1, 1)
    assert labels[slot] == "Re M2[l=0,k=2;r=0,s=1,b=1]"
    assert labels[slot + 50] == "Im M2[l=0,k=2;r=0,s=1,b=1]"
    assert len(set(labels)) == 154


def test_h0_ext_rows_222(index_222):
    _, ext, _ = oracle_rows(2, 2, 2)
    assert index_222.h0_ext == tuple(ext)
    assert all(row[4] >= 1 for row in index_222.h0_ext)
    assert all((row[0] + row[1]) % 2 == 0 for row in index_222.h0_ext)
    assert not set(index_222.h0_ext) & set(index_222.h0)


# ----------------------------------------------------------------------
# Oracle sweeps, including extreme truncations
# ----------------------------------------------------------------------

SWEEP = [
    (0, 0, 0, None),
    (0, 0, 2, None),
    (1, 1, 1, None),
    (2, 2, 2, None),
    (2, 2, 2, 3),
    (2, 2, 2, 0),
    (1, 2, 3, None),
    (3, 1, 0, None),
    (0, 5, 1, None),
    (8, 8, 2, None),
    (8, 8, 3, 4),
    (12, 0, 0, None),
    (4, 4, 6, None),
]


@pytest.mark.parametrize("L_mu,L_eta,N,depth", SWEEP)
def test_rows_match_oracle(L_mu, L_eta, N, depth):
    idx = MomentIndex.build(Truncation(L_mu, L_eta, N, depth_degree=depth))
    h0, ext, h2 = oracle_rows(L_mu, L_eta, N, depth)
    assert idx.h0 == tuple(h0)
    assert idx.h0_ext == tuple(ext)
    assert idx.h2 == tuple(h2)
    n0, n2, n_real = oracle_sizes(L_mu, L_eta, N, depth)
    assert (idx.n0, idx.n2, idx.n_real) == (n0, n2, n_real)
    assert idx.n_lk == (L_mu + 1) * (L_eta + 1)
    assert idx.pairs == tuple(sorted(set((r[0], r[1]) for r in h0)))


@pytest.mark.parametrize("L_mu", range(0, 7))
@pytest.mark.parametrize("L_eta", range(0, 7))
def test_n_plus_formula_with_d(L_mu, L_eta):
    idx = MomentIndex.build(Truncation(L_mu, L_eta, 0))
    d = 1 if (L_mu % 2 == 0 and L_eta % 2 == 0) else 0
    n_plus = ((L_mu + 1) * (L_eta + 1) + d) // 2
    n_minus = ((L_mu + 1) * (L_eta + 1) - d) // 2
    assert idx.n2 == n_plus  # C(3,3) = 1 at N = 0
    assert sum(idx.parity_I()) == n_plus
    assert sum(idx.parity_V()) == n_minus
    assert ((L_mu + 1) * (L_eta + 1) + d) % 2 == 0


@pytest.mark.parametrize("L_mu,L_eta,N,depth", SWEEP)
def test_positions_and_entries_are_consistent(L_mu, L_eta, N, depth):
    idx = MomentIndex.build(Truncation(L_mu, L_eta, N, depth_degree=depth))
    entries = idx.entries()
    assert len(entries) == idx.n_real
    assert all(isinstance(e, Entry) for e in entries)
    assert [e.slot for e in entries] == list(range(idx.n_real))
    assert [e.label for e in entries] == list(idx.labels())
    for e in entries:
        pos = idx.position(e.h, e.l, e.k, e.r, e.s, e.b)
        if e.part == "im":
            assert e.slot == pos + idx.n2
        else:
            assert e.slot == pos
        assert e.part == ("real" if e.h == 0 else e.part)
    parts = [e.part for e in entries]
    assert parts == ["real"] * idx.n0 + ["re"] * idx.n2 + ["im"] * idx.n2
    h_of = [e.h for e in entries]
    assert h_of == [0] * idx.n0 + [2] * (2 * idx.n2)


# ----------------------------------------------------------------------
# Parity masks, pair maps, exponents
# ----------------------------------------------------------------------


def test_parity_masks_222(index_222):
    pI = index_222.parity_I()
    pV = index_222.parity_V()
    assert len(pI) == len(pV) == 54
    assert all(a != b for a, b in zip(pI, pV))
    for (l, k, r, s, b), a in zip(index_222.h0, pI):
        assert a == ((l + k) % 2 == 0)
    assert sum(pI) == 5 * 6 and sum(pV) == 4 * 6
    assert pI[0] is True and pV[0] is False


def test_pair_maps_222(index_222):
    h0p = index_222.h0_pairs()
    h2p = index_222.h2_pairs()
    assert len(h0p) == 54 and len(h2p) == 50
    for row, p in zip(index_222.h0, h0p):
        assert index_222.pairs[p] == (row[0], row[1])
    for row, p in zip(index_222.h2, h2p):
        assert index_222.pairs[p] == (row[0], row[1])
        assert (row[0] + row[1]) % 2 == 0
    assert index_222.pairs == tuple(itertools.product(range(3), range(3)))


def test_exponents_helper(index_222):
    arr = index_222.exponents(index_222.h2)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (50, 5) and arr.dtype.kind == "i"
    assert arr.tolist() == [list(row) for row in index_222.h2]
    empty = index_222.exponents(())
    assert empty.shape == (0, 5)
    assert index_222.exponents([(1, 2, 3, 4, 5)]).tolist() == [[1, 2, 3, 4, 5]]


# ----------------------------------------------------------------------
# Parity switch and components (both sides of every dispatch)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("L_mu,L_eta,N", [(1, 1, 1), (2, 2, 2), (3, 2, 1), (0, 0, 2)])
def test_parity_false_keeps_odd_h2_rows(L_mu, L_eta, N):
    on = MomentIndex.build(Truncation(L_mu, L_eta, N), parity=True)
    off = MomentIndex.build(Truncation(L_mu, L_eta, N), parity=False)
    h0, ext, h2 = oracle_rows(L_mu, L_eta, N, parity=False)
    assert off.h2 == tuple(h2)
    assert off.h0 == on.h0 == tuple(h0)
    assert off.n2 == (L_mu + 1) * (L_eta + 1) * math.comb(N + 3, 3)
    assert set(on.h2) < set(off.h2) or (L_mu == L_eta == 0)
    assert off.h0_ext == tuple(ext)
    assert off.parity is False and on.parity is True
    assert hash(off) != hash(on)
    for row in off.h2:
        assert off.h2_pairs()[off.h2.index(row)] == off.pairs.index((row[0], row[1]))


def test_components_without_V_drops_odd_h0_rows(index_222):
    idx = MomentIndex.build(Truncation(2, 2, 2), components=("I", "Q"))
    h0, ext, h2 = oracle_rows(2, 2, 2, with_V=False)
    assert idx.h0 == tuple(h0)
    assert idx.h2 == tuple(h2) == index_222.h2
    assert idx.n0 == 5 * 6 and idx.n2 == 50 and idx.n_real == 30 + 100
    assert all(idx.parity_I()) and not any(idx.parity_V())
    assert idx.pairs == ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
    assert idx.n_lk == 5
    assert idx.position(0, 0, 0, 0, 0, 0) == 0
    with pytest.raises(ValueError):
        idx.position(0, 0, 1, 0, 0, 0)
    assert idx != index_222 and hash(idx) != hash(index_222)


def test_components_without_V_and_parity_false():
    idx = MomentIndex.build(Truncation(1, 1, 1), parity=False, components=("I", "Q"))
    assert idx.n0 == 2 * 3
    assert idx.n2 == 4 * 4
    assert idx.pairs == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert set(idx.h0_pairs()) == {0, 3}
    assert set(idx.h2_pairs()) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "components",
    [(), ("Q",), ("I", "X"), ("I", "I"), "IQ", ("I", 1)],
)
def test_components_validation(components):
    with pytest.raises(ValueError):
        MomentIndex.build(Truncation(1, 1, 1), components=components)


def test_parity_must_be_bool():
    with pytest.raises(ValueError):
        MomentIndex.build(Truncation(1, 1, 1), parity=1)
    with pytest.raises(ValueError):
        MomentIndex.build((1, 1, 1))


# ----------------------------------------------------------------------
# position() error paths
# ----------------------------------------------------------------------


def test_position_rejects_rows_outside_m(index_222):
    with pytest.raises(ValueError):
        index_222.position(1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        index_222.position(0, 0, 0, 0, 0, 1)  # h0_ext row, not in m
    with pytest.raises(ValueError):
        index_222.position(0, 3, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        index_222.position(0, 0, 0, 2, 1, 0)  # r + s > N
    with pytest.raises(ValueError):
        index_222.position(2, 0, 1, 0, 0, 0)  # odd parity dropped
    with pytest.raises(ValueError):
        index_222.position(2, 0, 0, 1, 1, 1)  # r + s + b > N
    with pytest.raises(ValueError):
        index_222.position(2, 0, 0, 0, 0, -1)


def test_ext_position_222(index_222):
    for i, row in enumerate(index_222.h0_ext):
        assert index_222.ext_position(*row) == i
    with pytest.raises(ValueError):
        index_222.ext_position(0, 0, 0, 0, 0)


# ----------------------------------------------------------------------
# Hashability, immutability, JIT static use
# ----------------------------------------------------------------------


def test_index_hashable_equal_and_leafless(index_222, truncation_222):
    again = MomentIndex.build(truncation_222)
    assert hash(index_222) == hash(again)
    assert index_222 == again
    assert jax.tree_util.tree_leaves(index_222) == []
    assert {index_222: 1}[again] == 1


def test_index_is_immutable(index_222):
    with pytest.raises((AttributeError, TypeError)):
        index_222.n0 = 3  # type: ignore[misc]


def test_index_usable_as_static_field_in_jit(index_222, index_111):
    class Holder(eqx.Module):
        index: MomentIndex = eqx.field(static=True)
        m: jax.Array

    compiled_calls = []

    @jax.jit
    def f(holder):
        compiled_calls.append(holder.index.n_real)
        mask = jnp.asarray(holder.index.parity_I(), dtype=float)
        return jnp.sum(holder.m[: holder.index.n0] * mask)

    m = jnp.arange(154.0)
    out = f(Holder(index_222, m))
    expected = sum(float(i) for i, a in enumerate(index_222.parity_I()) if a)
    assert float(out) == expected
    f(Holder(index_222, m + 1.0))
    assert compiled_calls == [154]  # one compile per static index
    f(Holder(index_111, jnp.arange(28.0)))
    assert compiled_calls == [154, 28]


def test_index_static_arg_of_jit(index_222):
    def g(m, index):
        return m[index.position(2, 0, 2, 0, 1, 0)]

    m = jnp.arange(154.0)
    assert float(jax.jit(g, static_argnums=1)(m, index_222)) == 67.0
