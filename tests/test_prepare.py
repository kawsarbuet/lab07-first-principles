"""Guards for splitting, leakage, Dataset types and DataLoader ordering."""

import numpy as np
import torch

from prepare import NUMERIC, BikeDataset, make_loaders, prepare_data


def test_split_sizes_and_no_overlap(prepared):
    train = set(prepared.train_frame.index)
    valid = set(prepared.valid_frame.index)
    test = set(prepared.test_frame.index)
    assert (len(train), len(valid), len(test)) == (3600, 1200, 1200)
    assert not (train & valid or train & test or valid & test)
    assert len(train | valid | test) == 6000


def test_dataset_dtype_and_shapes(prepared):
    dataset = BikeDataset(prepared.x_train, prepared.y_train)
    features, target = dataset[0]
    assert len(dataset) == 3600
    assert features.dtype == target.dtype == torch.float32
    assert features.shape == (prepared.x_train.shape[1],)
    assert target.shape == (1,)


def test_train_only_scaler(prepared):
    count = len(NUMERIC)
    np.testing.assert_allclose(prepared.x_train[:, :count].mean(axis=0), 0, atol=1e-5)
    assert np.max(abs(prepared.x_valid[:, :count].mean(axis=0))) > 0.01
    scaler = prepared.transformer.named_transformers_["numeric"]
    np.testing.assert_allclose(
        scaler.mean_, prepared.train_frame[NUMERIC].to_numpy().mean(axis=0)
    )


def test_train_shuffle_repeatable_and_validation_ordered(prepared):
    a_train, a_valid, _ = make_loaders(prepared, batch_size=16, seed=77)
    b_train, b_valid, _ = make_loaders(prepared, batch_size=16, seed=77)
    assert torch.equal(next(iter(a_train))[0], next(iter(b_train))[0])
    assert torch.equal(next(iter(a_valid))[0], next(iter(b_valid))[0])
    assert not torch.equal(
        next(iter(a_train))[0], torch.from_numpy(prepared.x_train[:16])
    )


def test_repeatable_data_generation():
    a = prepare_data(seed=2026, n=100)
    b = prepare_data(seed=2026, n=100)
    np.testing.assert_array_equal(a.x_train, b.x_train)
