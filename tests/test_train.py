"""Guard training, memorization, evaluation and best-weight restoration."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import BikeNet
from prepare import BikeDataset, make_loaders
from train import fit_network, run_epoch, seed_everything


def test_one_batch_can_be_memorized():
    seed_everything(3)
    features = torch.randn((12, 8))
    target = (features[:, :1] * 1.4 + 3).clone()
    loader = DataLoader(BikeDataset(features.numpy(), target.numpy()), batch_size=12)
    model = BikeNet(8, hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    loss_fn = nn.MSELoss()
    first, _ = run_epoch(model, loader, loss_fn)
    for _ in range(350):
        run_epoch(model, loader, loss_fn, optimizer)
    last, _ = run_epoch(model, loader, loss_fn)
    assert last < first * 0.01


def test_validation_epoch_does_not_change_weights(prepared):
    _, valid, _ = make_loaders(prepared, batch_size=64)
    model = BikeNet(prepared.x_train.shape[1])
    before = copy.deepcopy(model.state_dict())
    loss, mae = run_epoch(model, valid, nn.MSELoss())
    assert loss >= 0 and mae >= 0
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )


def test_fit_history_and_best_weights_are_restored(prepared):
    seed_everything(42)
    train, valid, _ = make_loaders(prepared, batch_size=256, seed=42)
    model = BikeNet(prepared.x_train.shape[1], hidden=16)
    history, best_epoch = fit_network(
        model, train, valid, epochs=5, learning_rate=0.002
    )
    assert list(history.columns) == [
        "epoch",
        "train_loss",
        "train_mae",
        "val_loss",
        "val_mae",
    ]
    assert len(history) == 5
    assert best_epoch == int(history.loc[history.val_loss.idxmin(), "epoch"])
    restored_loss, _ = run_epoch(model, valid, nn.MSELoss())
    np.testing.assert_allclose(restored_loss, history.val_loss.min(), rtol=1e-6)


def test_epoch_fails_if_prediction_shape_is_wrong():
    dataset = BikeDataset(np.ones((4, 3), dtype=np.float32), np.ones(4))
    loader = DataLoader(dataset, batch_size=4)
    with torch.no_grad():
        model = nn.Linear(3, 2)
        try:
            run_epoch(model, loader, nn.MSELoss())
        except ValueError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("Expected explicit prediction-target shape guard")
