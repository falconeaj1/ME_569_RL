"""Train a small offline Connect Four value network from generated histories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from connect4.value_model import ValueNetwork, encode_arrays


def train_model(
    dataset_path: Path,
    output_path: Path,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
) -> None:
    with np.load(dataset_path, allow_pickle=False) as data:
        x = encode_arrays(data["boards"], data["current_players"])
        y = data["values_current_player"].astype(np.float32).reshape(-1, 1)

    if len(x) == 0:
        raise ValueError("dataset contains no positions")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    validation_count = max(1, int(len(x) * validation_fraction)) if len(x) > 1 else 0
    val_idx = indices[:validation_count]
    train_idx = indices[validation_count:]
    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    model = ValueNetwork.initialize(hidden_size=hidden_size, seed=seed)
    adam_state = {
        "mw1": np.zeros_like(model.w1),
        "vw1": np.zeros_like(model.w1),
        "mb1": np.zeros_like(model.b1),
        "vb1": np.zeros_like(model.b1),
        "mw2": np.zeros_like(model.w2),
        "vw2": np.zeros_like(model.w2),
        "mb2": np.zeros_like(model.b2),
        "vb2": np.zeros_like(model.b2),
    }
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    step = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_idx)
        for start_idx in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start_idx : start_idx + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]

            hidden = np.tanh(xb @ model.w1 + model.b1)
            pred = np.tanh(hidden @ model.w2 + model.b2)
            grad_pred = (2.0 / len(xb)) * (pred - yb)
            grad_z2 = grad_pred * (1.0 - pred**2)
            grad_w2 = hidden.T @ grad_z2
            grad_b2 = grad_z2.sum(axis=0)
            grad_hidden = grad_z2 @ model.w2.T
            grad_z1 = grad_hidden * (1.0 - hidden**2)
            grad_w1 = xb.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            step += 1
            for name, param, grad in [
                ("w1", model.w1, grad_w1),
                ("b1", model.b1, grad_b1),
                ("w2", model.w2, grad_w2),
                ("b2", model.b2, grad_b2),
            ]:
                m_key = f"m{name}"
                v_key = f"v{name}"
                adam_state[m_key] = beta1 * adam_state[m_key] + (1.0 - beta1) * grad
                adam_state[v_key] = beta2 * adam_state[v_key] + (1.0 - beta2) * (grad**2)
                m_hat = adam_state[m_key] / (1.0 - beta1**step)
                v_hat = adam_state[v_key] / (1.0 - beta2**step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        train_loss = mean_squared_error(model.predict_features(x[train_idx]), y[train_idx].reshape(-1))
        if len(val_idx):
            val_pred = model.predict_features(x[val_idx])
            val_loss = mean_squared_error(val_pred, y[val_idx].reshape(-1))
            val_sign_accuracy = sign_accuracy(val_pred, y[val_idx].reshape(-1))
        else:
            val_loss = float("nan")
            val_sign_accuracy = float("nan")
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "val_sign_accuracy": val_sign_accuracy})
        print(
            f"epoch {epoch:>3}: train_mse={train_loss:.4f} val_mse={val_loss:.4f} "
            f"val_sign_acc={val_sign_accuracy:.2%}",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    model.metadata.update(
        {
            "dataset": str(dataset_path),
            "positions": int(len(x)),
            "train_positions": int(len(train_idx)),
            "validation_positions": int(len(val_idx)),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "history": history,
        }
    )
    model.save(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(model.metadata, indent=2))
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")


def mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def sign_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    mask = target != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(target[mask])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy value network from Connect Four histories.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/value_data/connect4_value_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/value_models/connect4_value_model.npz"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        dataset_path=args.dataset,
        output_path=args.output,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
