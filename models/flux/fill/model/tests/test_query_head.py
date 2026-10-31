"""QueryHead: every query form gives an embedding, a read gives half-hourly distributions, and a place-only embedding
does not change with the context."""

import torch

from data import V_TAB
from net import QueryHead


def forms():
    site = torch.tensor([[0.5, 0.5, 0.7, 0.2]])
    day, var = torch.randn(1, 3, 5), torch.tensor([[0, 3, 7]])
    return site, day, var, torch.randn(1, 40, 32)


def test_every_form_and_read():
    q = QueryHead(32, 16)
    site, day, var, h = forms()
    for ud, uv, uc in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)):
        f = lambda x: torch.full((1, 3), bool(x))
        z = q.embed(site, day, var, f(ud), f(uv), h, f(uc))
        assert z.shape == (1, 3, 16)
        loc, scale, df = q.read(z, day, var)
        assert loc.shape == (1, 3, 48) and (scale > 0).all() and (df > 2).all()


def test_absent_parts_do_not_enter():
    q = QueryHead(32, 16)
    site, day, var, h = forms()
    no = torch.zeros(1, 3, dtype=torch.bool)
    a = q.embed(site, day, var, no, no, h, no)
    b = q.embed(site, torch.randn(1, 3, 5), torch.tensor([[1, 2, 4]]), no, no, torch.randn(1, 40, 32), no)
    assert torch.equal(a, b)


def test_context_enters_when_used():
    q = QueryHead(32, 16)
    site, day, var, h = forms()
    yes = torch.ones(1, 3, dtype=torch.bool)
    assert not torch.equal(q.embed(site, day, var, yes, yes, h, yes), q.embed(site, day, var, yes, yes, torch.randn(1, 40, 32), yes))
    assert V_TAB > 3
