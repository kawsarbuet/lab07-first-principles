"""Split first; fit preprocessing on training rows; build float32 Datasets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
SEED = 2026


class BikeDataset(Dataset):
    """One transformed station-hour per request; both tensors are float32."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            raise ValueError("Feature and target lengths must match")
        self.features = torch.as_tensor(
            np.asarray(features, dtype=np.float32).copy(), dtype=torch.float32
        )
        self.targets = torch.as_tensor(
            np.asarray(targets, dtype=np.float32).copy(), dtype=torch.float32
        ).reshape(-1, 1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@dataclass
class Prepared:
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    transformer: ColumnTransformer
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


def prepare_data(seed: int = SEED, n: int = 6000) -> Prepared:
    """Create disjoint 60/20/20 splits and train-only feature preparation."""
    frame = make_rentals(n=n, seed=seed)
    train, remaining = train_test_split(frame, test_size=0.4, random_state=seed)
    valid, test = train_test_split(remaining, test_size=0.5, random_state=seed)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        sparse_threshold=0,
    )
    x_train = preprocessor.fit_transform(train).astype(np.float32)
    x_valid = preprocessor.transform(valid).astype(np.float32)
    x_test = preprocessor.transform(test).astype(np.float32)
    return Prepared(
        train_frame=train,
        valid_frame=valid,
        test_frame=test,
        transformer=preprocessor,
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=train[TARGET].to_numpy(dtype=np.float32),
        y_valid=valid[TARGET].to_numpy(dtype=np.float32),
        y_test=test[TARGET].to_numpy(dtype=np.float32),
    )


def make_loaders(
    data: Prepared, batch_size: int = 128, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Only train batches shuffle; seed the sampler for reproducibility."""
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        BikeDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid = DataLoader(
        BikeDataset(data.x_valid, data.y_valid), batch_size=batch_size, shuffle=False
    )
    test = DataLoader(
        BikeDataset(data.x_test, data.y_test), batch_size=batch_size, shuffle=False
    )
    return train, valid, test
