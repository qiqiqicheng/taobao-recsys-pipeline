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


def process_dt(df):
    dt_series = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = dt_series.dt.hour + 1
    df["day_of_week"] = dt_series.dt.dayofweek + 1
    df["month_of_year"] = dt_series.dt.month + 1
    df["is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

    # delta time
    df = df.sort_values(by=["user_id", "timestamp"])
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["delta_time"] = df["timestamp"] - df["prev_timestamp"]
    # fillna with median
    df["delta_time"] = df["delta_time"].fillna(df["delta_time"].median())
    # bins
    num_delta_time_bins = 20
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
    df = process_dt(df)

    # convert user_id
    user_id_2_idx = {int(user_id): idx + 1 for idx, user_id in enumerate(df["user_id"].unique())}
    idx_2_user_id = {idx: user_id for user_id, idx in user_id_2_idx.items()}
    df["user_id"] = df["user_id"].map(user_id_2_idx)
    with open(ROOT_DIR / "data/processed/maps/user_id_2_idx.json", "w") as f:
        json.dump(user_id_2_idx, f)
    with open(ROOT_DIR / "data/processed/maps/idx_2_user_id.json", "w") as f:
        json.dump(idx_2_user_id, f)

    # convert item_id
    item_id_2_idx = {int(item_id): idx + 1 for idx, item_id in enumerate(df["item_id"].unique())}
    idx_2_item_id = {idx: item_id for item_id, idx in item_id_2_idx.items()}
    df["item_id"] = df["item_id"].map(item_id_2_idx)
    with open(ROOT_DIR / "data/processed/maps/item_id_2_idx.json", "w") as f:
        json.dump(item_id_2_idx, f)
    with open(ROOT_DIR / "data/processed/maps/idx_2_item_id.json", "w") as f:
        json.dump(idx_2_item_id, f)

    # convert cat_id
    cat_id_2_idx = {int(cat_id): idx + 1 for idx, cat_id in enumerate(df["cat_id"].unique())}
    idx_2_cat_id = {idx: cat_id for cat_id, idx in cat_id_2_idx.items()}
    df["cat_id"] = df["cat_id"].map(cat_id_2_idx)
    with open(ROOT_DIR / "data/processed/maps/cat_id_2_idx.json", "w") as f:
        json.dump(cat_id_2_idx, f)
    with open(ROOT_DIR / "data/processed/maps/idx_2_cat_id.json", "w") as f:
        json.dump(idx_2_cat_id, f)

    # convert action
    action_2_idx = {action: idx + 1 for idx, action in enumerate(df["action"].unique())}
    idx_2_action = {idx: action for action, idx in action_2_idx.items()}
    df["action"] = df["action"].map(action_2_idx)
    with open(ROOT_DIR / "data/processed/maps/action_2_idx.json", "w") as f:
        json.dump(action_2_idx, f)
    with open(ROOT_DIR / "data/processed/maps/idx_2_action.json", "w") as f:
        json.dump(idx_2_action, f)

    # aggregate user data
    user_df = aggregate_user_data(df)

    # save user data
    user_df.to_csv(ROOT_DIR / "data/processed/user_data.csv", index=False)
    print(f"User data saved to {ROOT_DIR / 'data/processed/user_data.csv'}")

    end = time.time()
    print(f"Data preparation completed in {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
