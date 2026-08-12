import torch

from deepearth.core.fusion import DeepEarth


def test_family_factorization_preserves_species_ratios_and_matches_family_posterior():
    model = object.__new__(DeepEarth)
    torch.nn.Module.__init__(model)
    model.register_buffer("species_family", torch.tensor([0, 0, 1, 1]))
    model.family_count = 2

    species = torch.tensor([[1.0, 0.0, 2.0, -1.0]])
    family = torch.tensor([[-2.0, 2.0]])
    before = species.softmax(-1)
    after = model._factor_family_mass(species, family).softmax(-1)

    assert torch.allclose(after[:, :2].sum(-1), family.softmax(-1)[:, 0])
    assert torch.allclose(after[:, 2:].sum(-1), family.softmax(-1)[:, 1])
    assert torch.allclose(after[:, 0] / after[:, 1], before[:, 0] / before[:, 1])
    assert torch.allclose(after[:, 2] / after[:, 3], before[:, 2] / before[:, 3])


def test_family_loss_does_not_update_alphaearth_features():
    model = object.__new__(DeepEarth)
    torch.nn.Module.__init__(model)
    model.type_emb = torch.nn.Parameter(torch.zeros(1, 4))
    model.species_variable = "identity"
    model.register_buffer("species_family", torch.tensor([0, 0, 1, 1]))
    model.family_count = 2
    model.family_ae_head = torch.nn.Linear(64, 2)
    alphaearth = torch.randn(3, 64, requires_grad=True)

    loss = model.family_alphaearth_loss(
        {"identity": torch.tensor([0, 2, 3]), "alphaearth": alphaearth},
        {"identity": torch.ones(3, dtype=torch.bool), "alphaearth": torch.ones(3, dtype=torch.bool)},
    )
    loss.backward()

    assert alphaearth.grad is None
    assert model.family_ae_head.weight.grad is not None
