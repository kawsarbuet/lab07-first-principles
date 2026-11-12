"""Own epoch/fit loop, validation-selected experiments, baselines and reports."""

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn

from model import BikeNet
from prepare import SEED, make_loaders, prepare_data

REPORTS = Path(__file__).resolve().parent / "reports"


def seed_everything(seed: int = SEED) -> None:
    """Seed all randomness; use deterministic CPU operations and one thread."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Train if an optimizer is provided; otherwise measure without updates."""
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    absolute_error = 0.0
    seen = 0
    for features, target in loader:
        with torch.set_grad_enabled(training):
            prediction = model(features)
            if prediction.shape != target.shape:
                raise ValueError("Prediction and target must both have shape (N, 1)")
            loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        batch_n = len(target)
        running_loss += float(loss.item()) * batch_n
        absolute_error += float(torch.abs(prediction - target).sum().item())
        seen += batch_n
    return running_loss / seen, absolute_error / seen


def fit_network(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    *,
    epochs: int = 100,
    learning_rate: float = 0.003,
) -> tuple[pd.DataFrame, int]:
    """Log each epoch and restore the lowest-validation-loss weights."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        valid_loss, valid_mae = run_epoch(model, valid_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": valid_loss,
                "val_mae": valid_mae,
            }
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    return pd.DataFrame(history), best_epoch


def predict_network(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features)).numpy().reshape(-1)


def plot_curves(history: pd.DataFrame, best_epoch: int, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Training MSE")
    ax.plot(history["epoch"], history["val_loss"], label="Validation MSE")
    ax.axvline(best_epoch, ls="--", color="black", label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE (rentals²)", title="Bike demand learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def run_experiment(
    name: str,
    data,
    *,
    hidden: int = 64,
    learning_rate: float = 0.003,
    epochs: int = 100,
) -> tuple[dict, pd.DataFrame, BikeNet]:
    seed_everything()
    train_loader, valid_loader, _ = make_loaders(data)
    model = BikeNet(data.x_train.shape[1], hidden=hidden)
    history, best_epoch = fit_network(
        model,
        train_loader,
        valid_loader,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    best_row = history.iloc[best_epoch - 1]
    record = {
        "run": name,
        "hidden": hidden,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_mae": float(best_row["val_mae"]),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_val_loss": float(history.iloc[-1]["val_loss"]),
    }
    return record, history, model


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    seed_everything()
    data = prepare_data()
    print("Split sizes:", len(data.y_train), len(data.y_valid), len(data.y_test))
    print("First five rows:\n", data.train_frame.head().to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(data.train_frame["hour"], data.train_frame["rentals"], s=4, alpha=0.2)
    ax.set(xlabel="Hour (0–23)", ylabel="Rentals", title="Rentals by hour (train)")
    fig.tight_layout()
    fig.savefig(REPORTS / "rentals_by_hour.png", dpi=140)
    plt.close(fig)

    configs = [
        ("main", 64, 0.003),
        ("low_learning_rate", 64, 0.00003),
        ("wide_network", 256, 0.003),
    ]
    records = []
    saved_models = {}
    for name, hidden, rate in configs:
        record, history, model = run_experiment(
            name, data, hidden=hidden, learning_rate=rate
        )
        records.append(record)
        saved_models[name] = model
        history.to_csv(REPORTS / f"history_{name}.csv", index=False)
        if name == "main":
            history.to_csv(REPORTS / "history.csv", index=False)
        curve_name = "learning_curves.png" if name == "main" else f"curves_{name}.png"
        plot_curves(history, record["best_epoch"], REPORTS / curve_name)
        print(
            name, "best epoch", record["best_epoch"],
            "val MAE", round(record["best_val_mae"], 3),
        )
    pd.DataFrame(records).to_csv(REPORTS / "experiments.csv", index=False)

    # Model selection uses ONLY validation scores. Test is accessed once below.
    best_record = min(records, key=lambda row: row["best_val_loss"])
    selected = saved_models[best_record["run"]]
    mean_prediction = float(data.y_train.mean())
    linear = LinearRegression().fit(data.x_train, data.y_train)
    predictions = {
        "training_mean": (
            np.full_like(data.y_valid, mean_prediction),
            np.full_like(data.y_test, mean_prediction),
        ),
        "linear_regression": (
            linear.predict(data.x_valid),
            linear.predict(data.x_test),
        ),
        "selected_network": (
            predict_network(selected, data.x_valid),
            predict_network(selected, data.x_test),
        ),
    }
    metrics = {
        "seed": SEED,
        "rows": 6000,
        "split": {"train": 3600, "validation": 1200, "test": 1200},
        "selected_run": best_record["run"],
        "selection_rule": "lowest best validation MSE; test used once after selection",
        "models": {
            name: {
                "validation_mae_rentals": float(
                    mean_absolute_error(data.y_valid, pred_valid)
                ),
                "test_mae_rentals": float(
                    mean_absolute_error(data.y_test, pred_test)
                ),
            }
            for name, (pred_valid, pred_test) in predictions.items()
        },
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("Final model selected using validation:", best_record["run"])
    print(json.dumps(metrics["models"], indent=2))
    print("Wrote reports/ files")


if __name__ == "__main__":
    main()
