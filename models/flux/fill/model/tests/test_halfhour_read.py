"""HalfHourRead and TokenConv: zero at initialisation, and the read sees only visible values."""

import torch

from data import V_TAB
from net import HalfHourRead, TokenConv


def test_read_is_zero_at_start_and_shaped():
    m = HalfHourRead(32, 8)
    z, obs = torch.randn(2, V_TAB, 180, 48), torch.rand(2, V_TAB, 180, 48) > 0.5
    e = m(torch.randn(2, V_TAB, 180, 32), z * obs, obs)
    assert e.shape == (2, V_TAB, 180, 48, 4) and e.abs().max() == 0


def test_read_ignores_hidden_values():
    m = HalfHourRead(32, 8)
    torch.nn.init.normal_(m.read.weight)
    h, z, obs = torch.randn(1, V_TAB, 180, 32), torch.randn(1, V_TAB, 180, 48), torch.rand(1, V_TAB, 180, 48) > 0.5
    z2 = torch.where(obs, z, torch.randn_like(z))
    assert torch.equal(m(h, z, obs), m(h, z2, obs))


def test_read_differs_by_half_hour():
    m = HalfHourRead(32, 8)
    torch.nn.init.normal_(m.read.weight)
    e = m(torch.randn(1, V_TAB, 180, 32), torch.randn(1, V_TAB, 180, 48), torch.ones(1, V_TAB, 180, 48, dtype=torch.bool))
    assert e[0, 0, 0, 0].abs().sum() > 0 and not torch.equal(e[0, 0, 0, 0], e[0, 0, 0, 1])


def test_token_conv_zero_at_start():
    m = TokenConv(5, 8, 32)
    out = m(torch.randn(2, V_TAB, 180, 5 * 48))
    assert out.shape == (2, V_TAB, 180, 32) and out.abs().max() == 0
