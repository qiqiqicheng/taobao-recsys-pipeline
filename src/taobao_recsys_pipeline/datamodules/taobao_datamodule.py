import ast
import os
import random
from typing import Any, Literal

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader

from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


def parse_seq(val) -> list:
    """Coerce a sequence column value to a plain Python list.

    Handles both legacy CSV format (string repr of list) and
    Parquet format (native list / numpy array).
    """
    if isinstance(val, str):
        return ast.literal_eval(val)
    return list(val)


def load_data(file: str | pd.DataFrame):
    if isinstance(file, pd.DataFrame):
        return file
    elif isinstance(file, str) and file.endswith(".csv"):
        return pd.read_csv(file)
    elif isinstance(file, str) and file.endswith(".parquet"):
        return pd.read_parquet(file)
    elif isinstance(file, str) and file.endswith(".pt"):
        return torch.load(file)
    else:
        raise ValueError("ratings_file must be a csv, parquet, or pt file")


def save_data(df: pd.DataFrame, file: str):
    if file.endswith(".csv"):
        df.to_csv(file, index=False)
        log.info(f"Data saved to {file}")
    else:
        raise ValueError("Only csv format is supported for saving data.")


class TaobaoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str | pd.DataFrame,
        max_seq_len: int,
        split: Literal["train", "val", "test"],
        min_seq_len: int = 5,
    ) -> None:
        super().__init__()
        df = load_data(file)
        self.df = df[df["item_id"].apply(lambda x: len(parse_seq(x)) >= min_seq_len)]
        self._cache = dict()
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Any:
        if index in self._cache:
            return self._cache[index]
        sample = self.load_sample(index)
        if self.split != "train":
            self._cache[index] = sample
        return sample

    def load_sample(self, index):
        row = self.df.iloc[index]
        user_id = row["user_id"]

        item_seq = parse_seq(row["item_id"])
        cat_seq = parse_seq(row["cat_id"])
        action_seq = parse_seq(row["action"])
        hour_of_day_seq = parse_seq(row["hour_of_day"])
        day_of_week_seq = parse_seq(row["day_of_week"])
        month_of_year_seq = parse_seq(row["month_of_year"])
        is_weekend_seq = parse_seq(row["is_weekend"])
        delta_time_seq = parse_seq(row["delta_time"])

        seq_len = len(item_seq)

        if self.split == "test":
            target_idx = seq_len - 1
        elif self.split == "val":
            target_idx = seq_len - 2
        elif self.split == "train":
            target_idx = random.randint(1, seq_len - 3)

        target_item = item_seq[target_idx]
        target_cat = cat_seq[target_idx]
        target_action = action_seq[target_idx]
        target_hour_of_day = hour_of_day_seq[target_idx]
        target_day_of_week = day_of_week_seq[target_idx]
        target_month_of_year = month_of_year_seq[target_idx]
        target_is_weekend = is_weekend_seq[target_idx]
        target_delta_time = delta_time_seq[target_idx]

        hist_item = item_seq[:target_idx]
        hist_cat = cat_seq[:target_idx]
        hist_action = action_seq[:target_idx]
        hist_hour_of_day = hour_of_day_seq[:target_idx]
        hist_day_of_week = day_of_week_seq[:target_idx]
        hist_month_of_year = month_of_year_seq[:target_idx]
        hist_is_weekend = is_weekend_seq[:target_idx]
        hist_delta_time = delta_time_seq[:target_idx]

        def truncate_or_pad(y: list[int], max_seq_len: int) -> list[int]:
            if len(y) > max_seq_len:
                return y[-max_seq_len:]
            else:
                return [0] * (max_seq_len - len(y)) + y  # NOTE: left padding with 0s

        hist_item_padded = truncate_or_pad(hist_item, self.max_seq_len)
        hist_cat_padded = truncate_or_pad(hist_cat, self.max_seq_len)
        hist_action_padded = truncate_or_pad(hist_action, self.max_seq_len)
        hist_hour_of_day_padded = truncate_or_pad(hist_hour_of_day, self.max_seq_len)
        hist_day_of_week_padded = truncate_or_pad(hist_day_of_week, self.max_seq_len)
        hist_month_of_year_padded = truncate_or_pad(hist_month_of_year, self.max_seq_len)
        hist_is_weekend_padded = truncate_or_pad(hist_is_weekend, self.max_seq_len)
        hist_delta_time_padded = truncate_or_pad(hist_delta_time, self.max_seq_len)

        sample_dict = {
            # "user_id": torch.tensor(user_id, dtype=torch.int32),
            "hist_item_id": torch.tensor(hist_item_padded, dtype=torch.int32),
            "hist_cat_id": torch.tensor(hist_cat_padded, dtype=torch.int32),
            "hist_action": torch.tensor(hist_action_padded, dtype=torch.int32),
            "hist_hour_of_day": torch.tensor(hist_hour_of_day_padded, dtype=torch.int32),
            "hist_day_of_week": torch.tensor(hist_day_of_week_padded, dtype=torch.int32),
            "hist_month_of_year": torch.tensor(hist_month_of_year_padded, dtype=torch.int32),
            "hist_is_weekend": torch.tensor(hist_is_weekend_padded, dtype=torch.int32),
            "hist_delta_time": torch.tensor(hist_delta_time_padded, dtype=torch.int32),
            "target_item_id": torch.tensor(target_item, dtype=torch.int32),
            "target_cat_id": torch.tensor(target_cat, dtype=torch.int32),
            "target_action": torch.tensor(target_action, dtype=torch.int32),
            "target_hour_of_day": torch.tensor(target_hour_of_day, dtype=torch.int32),
            "target_day_of_week": torch.tensor(target_day_of_week, dtype=torch.int32),
            "target_month_of_year": torch.tensor(target_month_of_year, dtype=torch.int32),
            "target_is_weekend": torch.tensor(target_is_weekend, dtype=torch.int32),
            "target_delta_time": torch.tensor(target_delta_time, dtype=torch.int32),
        }

        return sample_dict


class TaobaoDataModule(L.LightningDataModule):
    def __init__(
        self,
        file: str | pd.DataFrame,
        max_seq_len: int,
        batch_size: int,
        num_workers: int = os.cpu_count() // 4,  # type:ignore
    ) -> None:
        super().__init__()
        self.file = file
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TaobaoDataset(self.file, self.max_seq_len, split="train")
            self.val_dataset = TaobaoDataset(self.file, self.max_seq_len, split="val")

        if stage in (None, "test"):
            self.test_dataset = TaobaoDataset(self.file, self.max_seq_len, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # no shuffle for reproducibility
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # no shuffle for reproducibility
            num_workers=self.num_workers,
        )

    def save_predictions(self, output_file: str, predictions: dict):
        df = self.test_dataset.df
        for key, value in predictions.items():
            df[key] = value
        save_data(df, output_file)
