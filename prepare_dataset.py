import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    test_size: float
    val_size: float
    stratify: bool
    scale: bool


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size + val_size must be < 1, got {test_size + val_size}"
        )


def prepare_splits(
    *,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify: bool = True,
    scale: bool = True,
):
    """
    Loads sklearn's Breast Cancer Wisconsin dataset and produces reproducible splits.

    val_size is interpreted as a fraction of the remaining (train+val) pool after
    removing the test set.
    """
    _validate_sizes(test_size, val_size)

    ds = load_breast_cancer(as_frame=True)
    X = ds.data.to_numpy(dtype=np.float32, copy=True)
    y = ds.target.to_numpy(dtype=np.int64, copy=True)
    feature_names = list(ds.feature_names)
    target_names = list(ds.target_names)

    strat = y if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    strat_tv = y_trainval if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=seed,
        stratify=strat_tv,
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        X_val = scaler.transform(X_val).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    cfg = SplitConfig(
        seed=seed, test_size=test_size, val_size=val_size, stratify=stratify, scale=scale
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_names": target_names,
        "config": asdict(cfg),
        "scaler_mean": None if scaler is None else scaler.mean_.astype(float).tolist(),
        "scaler_scale": None if scaler is None else scaler.scale_.astype(float).tolist(),
    }


def _class_balance(y: np.ndarray) -> dict:
    uniq, cnt = np.unique(y, return_counts=True)
    total = int(cnt.sum())
    return {int(u): {"count": int(c), "share": float(c / total)} for u, c in zip(uniq, cnt)}


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare Breast Cancer Wisconsin splits.")
    p.add_argument("--out", type=str, default="data_processed", help="Output directory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of train+val pool reserved for validation.",
    )
    p.add_argument("--no-stratify", action="store_true", help="Disable stratified splits.")
    p.add_argument("--no-scale", action="store_true", help="Disable StandardScaler.")
    args = p.parse_args()

    pack = prepare_splits(
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify=not args.no_stratify,
        scale=not args.no_scale,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main arrays
    np.save(out_dir / "X_train.npy", pack["X_train"])
    np.save(out_dir / "y_train.npy", pack["y_train"])
    np.save(out_dir / "X_val.npy", pack["X_val"])
    np.save(out_dir / "y_val.npy", pack["y_val"])
    np.save(out_dir / "X_test.npy", pack["X_test"])
    np.save(out_dir / "y_test.npy", pack["y_test"])

    meta = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "n_samples_total": int(pack["X_train"].shape[0] + pack["X_val"].shape[0] + pack["X_test"].shape[0]),
        "n_features": int(pack["X_train"].shape[1]),
        "feature_names": pack["feature_names"],
        "target_names": pack["target_names"],
        "split_config": pack["config"],
        "class_balance": {
            "train": _class_balance(pack["y_train"]),
            "val": _class_balance(pack["y_val"]),
            "test": _class_balance(pack["y_test"]),
        },
        "scaler": {
            "type": None if pack["scaler_mean"] is None else "StandardScaler",
            "mean": pack["scaler_mean"],
            "scale": pack["scaler_scale"],
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote splits to: {out_dir.resolve()}")
    print(json.dumps(meta["class_balance"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

