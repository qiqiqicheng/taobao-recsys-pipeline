import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DEBUG = False


def prepare_paths():
    if not os.path.exists(ROOT_DIR / "data/"):
        raise FileNotFoundError("The data directory does not exist.")

    Path(ROOT_DIR / "data/processed/maps/").mkdir(parents=True, exist_ok=True)
    Path(ROOT_DIR / "data/processed/features/").mkdir(parents=True, exist_ok=True)
    print("Directory structure is ready.")


def get_data(rows=None) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(
        ROOT_DIR / "data/raw/UserBehavior.csv",
        nrows=rows,
        header=None,
        names=["user_id", "item_id", "cat_id", "action", "timestamp"],
        dtype={"user_id": int, "item_id": int, "cat_id": int, "action": str, "timestamp": int},
    )
    return df


def process_dt(df, num_delta_time_bins):
    dt_series = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = dt_series.dt.hour + 1
    df["day_of_week"] = dt_series.dt.dayofweek + 1
    df["month_of_year"] = dt_series.dt.month
    df["is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int) + 1

    # delta time
    df = df.sort_values(by=["user_id", "timestamp"])
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["delta_time"] = df["timestamp"] - df["prev_timestamp"]
    # fillna with median
    df["delta_time"] = df["delta_time"].fillna(df["delta_time"].median())
    # bins
    df["delta_time"], delta_time_egdes = pd.qcut(
        np.log(df["delta_time"] + 1), q=num_delta_time_bins, labels=False, retbins=True
    )
    df["delta_time"] = df["delta_time"] + 1

    # save the edges
    np.save(ROOT_DIR / "data/processed/maps/delta_time_edges.npy", delta_time_egdes)
    print(f"Delta time edges saved: {len(delta_time_egdes)} bins.")

    # drop ts
    df = df.drop(columns=["timestamp", "prev_timestamp"])

    return df


def aggregate_user_data(df):
    # aggregate by user_id
    agg_funcs = {
        "item_id": lambda x: list(x),
        "cat_id": lambda x: list(x),
        "action": lambda x: list(x),
        "hour_of_day": lambda x: list(x),
        "day_of_week": lambda x: list(x),
        "month_of_year": lambda x: list(x),
        "is_weekend": lambda x: list(x),
        "delta_time": lambda x: list(x),
    }
    user_df = df.groupby("user_id").agg(agg_funcs).reset_index()
    return user_df


def build_and_save_mapping(series: pd.Series, maps_dir: Path, name: str, cast_int: bool = True) -> tuple[dict, dict]:
    """Build a 1-based value→index mapping and its inverse, then persist both as JSON."""
    val_2_idx = {(int(val) if cast_int else val): idx + 1 for idx, val in enumerate(series.unique())}
    idx_2_val = {idx: val for val, idx in val_2_idx.items()}
    with open(maps_dir / f"{name}_2_idx.json", "w") as f:
        json.dump(val_2_idx, f)
    with open(maps_dir / f"idx_2_{name}.json", "w") as f:
        json.dump(idx_2_val, f)
    return val_2_idx, idx_2_val


def main():
    start = time.time()
    if DEBUG:
        rows = 5000
        print(f"DEBUG mode: Only processing {rows} rows.")
    else:
        rows = None
        print("Processing the entire dataset.")

    prepare_paths()

    df = get_data(rows=rows)

    # convert timestamp
    num_delta_time_bins = 20
    df = process_dt(df, num_delta_time_bins=num_delta_time_bins)

    maps_dir = ROOT_DIR / "data/processed/maps"

    # convert user_id, item_id, cat_id, action → 1-based indices
    user_id_2_idx, _ = build_and_save_mapping(df["user_id"], maps_dir, "user_id")
    df["user_id"] = df["user_id"].map(user_id_2_idx)

    item_id_2_idx, _ = build_and_save_mapping(df["item_id"], maps_dir, "item_id")
    df["item_id"] = df["item_id"].map(item_id_2_idx)

    cat_id_2_idx, _ = build_and_save_mapping(df["cat_id"], maps_dir, "cat_id")
    df["cat_id"] = df["cat_id"].map(cat_id_2_idx)

    action_2_idx, _ = build_and_save_mapping(df["action"], maps_dir, "action", cast_int=False)
    df["action"] = df["action"].map(action_2_idx)

    # aggregate user data
    user_df = aggregate_user_data(df)

    # save user data (Parquet preserves native list columns, avoids ast.literal_eval overhead at load time)
    user_df.to_parquet(ROOT_DIR / "data/processed/user_data.parquet", index=False)
    print(f"User data saved to {ROOT_DIR / 'data/processed/user_data.parquet'}")

    end = time.time()
    print(f"Data preparation completed in {end - start:.2f} seconds.")

    # construct features
    # user sequence features
    user_sequence_features = [
        {
            "name": "hist_item_id",
            "vocab_size": len(item_id_2_idx) + 1,
        },
        {
            "name": "hist_cat_id",
            "vocab_size": len(cat_id_2_idx) + 1,
        },
        {
            "name": "hist_action",
            "vocab_size": len(action_2_idx) + 1,
        },
        {
            "name": "hist_hour_of_day",
            "vocab_size": 25,
        },
        {
            "name": "hist_day_of_week",
            "vocab_size": 8,
        },
        {
            "name": "hist_month_of_year",
            "vocab_size": 13,
        },
        {
            "name": "hist_is_weekend",
            "vocab_size": 3,
        },
        {
            "name": "hist_delta_time",
            "vocab_size": num_delta_time_bins + 1,
            "edges_file": "data/processed/maps/delta_time_edges.npy",  # used for inference-time binning
        },
    ]
    with open(ROOT_DIR / "data/processed/features/user_sequence_features.json", "w") as f:
        json.dump(user_sequence_features, f, indent=4)

    # user sparse faetures
    # no user sparse features for this dataset

    # item sparse faetures
    item_sparse_features = [
        {
            "name": "item_id",
            "vocab_size": len(item_id_2_idx) + 1,
        },
        {
            "name": "cat_id",
            "vocab_size": len(cat_id_2_idx) + 1,
        },
    ]

    with open(ROOT_DIR / "data/processed/features/item_sparse_features.json", "w") as f:
        json.dump(item_sparse_features, f, indent=4)


if __name__ == "__main__":
    main()
