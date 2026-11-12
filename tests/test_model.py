"""Network architecture and gradient correctness."""

import pytest
import torch
from torch import nn

from model import BikeNet


@pytest.mark.parametrize("hidden", [8, 16, 64])
def test_network_output_shape(hidden):
    model = BikeNet(8, hidden=hidden)
    assert model(torch.zeros((7, 8))).shape == (7, 1)
    assert sum(isinstance(module, nn.ReLU) for module in model.modules()) >= 2


def test_optimizer_step_updates_every_linear_layer():
    torch.manual_seed(5)
    model = BikeNet(8, hidden=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [
        layer.weight.detach().clone()
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]
    prediction = model(torch.rand((24, 8)) + 0.2)
    target = torch.full((24, 1), 20.0)
    loss = nn.MSELoss()(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = [layer.weight for layer in model.modules() if isinstance(layer, nn.Linear)]
    assert all(not torch.equal(x, y) for x, y in zip(before, after, strict=True))


def test_too_few_layers_rejected():
    with pytest.raises(ValueError):
        BikeNet(8, layers=1)
