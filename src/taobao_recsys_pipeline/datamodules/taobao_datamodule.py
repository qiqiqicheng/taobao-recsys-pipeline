import ast
import os
import random
from collections import Counter
from typing import Any, Literal

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader

from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


def parse_seq(val) -> list:
    """Coerce a sequence column value to a plain Python list.

    Handles both legacy CSV format (string repr of list) and Parquet format
    (native list / numpy array).
    """

    if isinstance(val, str):
        return ast.literal_eval(val)
    return list(val)


def load_data(file: str | pd.DataFrame):
    if isinstance(file, pd.DataFrame):
        return file
    if isinstance(file, str) and file.endswith(".csv"):
        return pd.read_csv(file)
    if isinstance(file, str) and file.endswith(".parquet"):
        return pd.read_parquet(file)
    if isinstance(file, str) and file.endswith(".pt"):
        return torch.load(file)
    raise ValueError("ratings_file must be a csv, parquet, or pt file")


def save_data(df: pd.DataFrame, file: str):
    if file.endswith(".csv"):
        df.to_csv(file, index=False)
        log.info(f"Data saved to {file}")
    else:
        raise ValueError("Only csv format is supported for saving data.")


def _build_item_stats(df: pd.DataFrame) -> tuple[list[int], list[float], dict[int, int]]:
    """Build popularity sampling table and item->cat lookup from aggregated user DF."""

    item_freq: Counter[int] = Counter()
    item2cat: dict[int, int] = {}

    for _, row in df.iterrows():
        items = parse_seq(row["item_id"])
        cats = parse_seq(row["cat_id"])
        for it, ca in zip(items, cats, strict=False):
            if it == 0:
                continue
            item_freq[it] += 1
            if it not in item2cat:
                item2cat[it] = ca

    item_ids = list(item_freq.keys())
    weights = [float(item_freq[it]) for it in item_ids]
    return item_ids, weights, item2cat


def _load_recall_candidates(file: str) -> dict[int, list[int]]:
    """Load recall candidates from parquet/csv.

    Expected columns:
      - user_id: int
      - recall_item_ids: list[int] or string repr of list
    """

    if file.endswith(".parquet"):
        df = pd.read_parquet(file)
    elif file.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        raise ValueError("recall_candidates_file must be a .parquet or .csv")

    if "user_id" not in df.columns or "recall_item_ids" not in df.columns:
        raise ValueError("recall candidates file must contain columns: user_id, recall_item_ids")

    out: dict[int, list[int]] = {}
    for _, row in df.iterrows():
        uid = int(row["user_id"])
        out[uid] = [int(x) for x in parse_seq(row["recall_item_ids"])]
    return out


class TaobaoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str | pd.DataFrame,
        max_seq_len: int,
        split: Literal["train", "val", "test"],
        min_seq_len: int = 5,
        task: Literal["match", "ctr"] = "match",
        # list-wise CTR sampling
        listwise_k: int = 0,
        n1_recall: int = 50,
        n2_popular: int = 50,
        recall_candidates: dict[int, list[int]] | None = None,
        pop_item_ids: list[int] | None = None,
        pop_weights: list[float] | None = None,
        item2cat: dict[int, int] | None = None,
        sampler_seed: int | None = None,
    ) -> None:
        super().__init__()
        df = load_data(file)
        self.df = df[df["item_id"].apply(lambda x: len(parse_seq(x)) >= min_seq_len)]
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.split = split
        self.task = task

        self.listwise_k = listwise_k
        self.n1_recall = n1_recall
        self.n2_popular = n2_popular
        self.recall_candidates = recall_candidates or {}
        self.pop_item_ids = pop_item_ids or []
        self.pop_weights = pop_weights or []
        self.item2cat = item2cat or {}
        self.sampler_seed = sampler_seed

        if self.task == "ctr" and self.listwise_k <= 1:
            raise ValueError("For CTR list-wise training, listwise_k must be >= 2")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Any:
        if index in self._cache:
            return self._cache[index]
        sample = self.load_sample(index)
        if self.split != "train":
            self._cache[index] = sample
        return sample

    def _rng_for_index(self, index: int) -> random.Random:
        if self.split == "train" or self.sampler_seed is None:
            return random  # type: ignore
        return random.Random(self.sampler_seed + index)

    def _sample_popular_negatives(
        self,
        rng: random.Random,
        exclude: set[int],
        n: int,
    ) -> list[int]:
        if n <= 0:
            return []
        if not self.pop_item_ids:
            return []

        # weighted sampling with replacement, then unique-filter
        picked: list[int] = []
        attempts = 0
        while len(picked) < n and attempts < n * 20:
            it = rng.choices(self.pop_item_ids, weights=self.pop_weights, k=1)[0]
            attempts += 1
            if it in exclude or it == 0:
                continue
            exclude.add(it)
            picked.append(it)
        return picked

    def load_sample(self, index):
        row = self.df.iloc[index]
        user_id = int(row["user_id"])

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
        else:  # train
            target_idx = random.randint(1, seq_len - 3)

        target_item = int(item_seq[target_idx])
        target_cat = int(cat_seq[target_idx])
        target_action = int(action_seq[target_idx])
        target_hour_of_day = int(hour_of_day_seq[target_idx])
        target_day_of_week = int(day_of_week_seq[target_idx])
        target_month_of_year = int(month_of_year_seq[target_idx])
        target_is_weekend = int(is_weekend_seq[target_idx])
        target_delta_time = int(delta_time_seq[target_idx])

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
            return [0] * (max_seq_len - len(y)) + y  # NOTE: left padding with 0s

        sample_dict: dict[str, torch.Tensor] = {
            "user_id": torch.tensor(user_id, dtype=torch.int32),
            "hist_item_id": torch.tensor(truncate_or_pad(hist_item, self.max_seq_len), dtype=torch.int32),
            "hist_cat_id": torch.tensor(truncate_or_pad(hist_cat, self.max_seq_len), dtype=torch.int32),
            "hist_action": torch.tensor(truncate_or_pad(hist_action, self.max_seq_len), dtype=torch.int32),
            "hist_hour_of_day": torch.tensor(truncate_or_pad(hist_hour_of_day, self.max_seq_len), dtype=torch.int32),
            "hist_day_of_week": torch.tensor(truncate_or_pad(hist_day_of_week, self.max_seq_len), dtype=torch.int32),
            "hist_month_of_year": torch.tensor(
                truncate_or_pad(hist_month_of_year, self.max_seq_len), dtype=torch.int32
            ),
            "hist_is_weekend": torch.tensor(truncate_or_pad(hist_is_weekend, self.max_seq_len), dtype=torch.int32),
            "hist_delta_time": torch.tensor(truncate_or_pad(hist_delta_time, self.max_seq_len), dtype=torch.int32),
            "target_item_id": torch.tensor(target_item, dtype=torch.int32),
            "target_cat_id": torch.tensor(target_cat, dtype=torch.int32),
            "target_action": torch.tensor(target_action, dtype=torch.int32),
            "target_hour_of_day": torch.tensor(target_hour_of_day, dtype=torch.int32),
            "target_day_of_week": torch.tensor(target_day_of_week, dtype=torch.int32),
            "target_month_of_year": torch.tensor(target_month_of_year, dtype=torch.int32),
            "target_is_weekend": torch.tensor(target_is_weekend, dtype=torch.int32),
            "target_delta_time": torch.tensor(target_delta_time, dtype=torch.int32),
        }

        if self.task == "ctr":
            rng = self._rng_for_index(index)

            # 1) recall candidates (n1)
            recall_pool = [int(x) for x in self.recall_candidates.get(user_id, [])]
            recall_pool = [x for x in recall_pool if x != 0 and x != target_item]
            if len(recall_pool) > self.n1_recall:
                recall_sample = rng.sample(recall_pool, k=self.n1_recall)
            else:
                recall_sample = recall_pool

            # 2) popularity sample (n2)
            exclude = {target_item, 0}
            pop_sample = self._sample_popular_negatives(rng, exclude=set(exclude), n=self.n2_popular)

            # 3) union + final sample to 1:(k-1)
            union_neg = list({*recall_sample, *pop_sample} - {target_item, 0})
            rng.shuffle(union_neg)

            need = self.listwise_k - 1
            negs = union_neg[:need]

            # 4) if not enough, keep filling from popularity
            exclude2 = {target_item, 0, *negs}
            while len(negs) < need:
                extra = self._sample_popular_negatives(rng, exclude2, n=need - len(negs))
                if not extra:
                    break
                negs.extend(extra)

            # As a last resort (e.g. tiny debug set), allow repeats from pop list
            while len(negs) < need and self.pop_item_ids:
                it = rng.choice(self.pop_item_ids)
                if it != 0 and it != target_item:
                    negs.append(it)

            candidates = [target_item] + negs[:need]
            if len(candidates) < self.listwise_k:
                # Keep fixed K for dataloader collation; 0 is padding/unknown id.
                candidates.extend([0] * (self.listwise_k - len(candidates)))

            cand_cats = [target_cat] + [int(self.item2cat.get(it, 0)) for it in candidates[1:]]

            # One-positive list-wise labels: positive is always candidate index 0.
            labels = [1.0] + [0.0] * (self.listwise_k - 1)

            sample_dict["candidate_item_id"] = torch.tensor(candidates[: self.listwise_k], dtype=torch.int32)  # [K]
            sample_dict["candidate_cat_id"] = torch.tensor(cand_cats[: self.listwise_k], dtype=torch.int32)  # [K]
            sample_dict["label"] = torch.tensor(labels, dtype=torch.float32)  # [K]

        return sample_dict


class TaobaoDataModule(L.LightningDataModule):
    def __init__(
        self,
        file: str | pd.DataFrame,
        max_seq_len: int,
        batch_size: int,
        num_workers: int = os.cpu_count() // 4,  # type:ignore
        task: Literal["match", "ctr"] = "match",
        # list-wise CTR sampling
        listwise_k: int = 20,
        n1_recall: int = 50,
        n2_popular: int = 50,
        recall_candidates_file: str | None = None,
        sampler_seed: int | None = 728,
    ) -> None:
        super().__init__()
        self.file = file
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        log.info(f"Using num_workers={self.num_workers} for DataLoader")

        self.task = task
        self.listwise_k = listwise_k
        self.n1_recall = n1_recall
        self.n2_popular = n2_popular
        self.recall_candidates_file = recall_candidates_file
        self.sampler_seed = sampler_seed

        # Prepared in setup when task == "ctr"
        self._recall_candidates: dict[int, list[int]] | None = None
        self._pop_item_ids: list[int] | None = None
        self._pop_weights: list[float] | None = None
        self._item2cat: dict[int, int] | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "test"):
            return

        df = load_data(self.file)

        if self.task == "ctr":
            item_ids, weights, item2cat = _build_item_stats(df)
            self._pop_item_ids = item_ids
            self._pop_weights = weights
            self._item2cat = item2cat

            if self.recall_candidates_file and os.path.exists(self.recall_candidates_file):
                self._recall_candidates = _load_recall_candidates(self.recall_candidates_file)
                log.info(f"Loaded recall candidates from {self.recall_candidates_file}")
            else:
                self._recall_candidates = {}
                if self.recall_candidates_file:
                    log.warning(
                        f"recall_candidates_file '{self.recall_candidates_file}' not found; "
                        "CTR training will fallback to popularity sampling only."
                    )

        if stage in (None, "fit"):
            self.train_dataset = TaobaoDataset(
                df,
                self.max_seq_len,
                split="train",
                task=self.task,
                listwise_k=self.listwise_k,
                n1_recall=self.n1_recall,
                n2_popular=self.n2_popular,
                recall_candidates=self._recall_candidates,
                pop_item_ids=self._pop_item_ids,
                pop_weights=self._pop_weights,
                item2cat=self._item2cat,
                sampler_seed=self.sampler_seed,
            )
            self.val_dataset = TaobaoDataset(
                df,
                self.max_seq_len,
                split="val",
                task=self.task,
                listwise_k=self.listwise_k,
                n1_recall=self.n1_recall,
                n2_popular=self.n2_popular,
                recall_candidates=self._recall_candidates,
                pop_item_ids=self._pop_item_ids,
                pop_weights=self._pop_weights,
                item2cat=self._item2cat,
                sampler_seed=self.sampler_seed,
            )

        if stage in (None, "test"):
            self.test_dataset = TaobaoDataset(
                df,
                self.max_seq_len,
                split="test",
                task=self.task,
                listwise_k=self.listwise_k,
                n1_recall=self.n1_recall,
                n2_popular=self.n2_popular,
                recall_candidates=self._recall_candidates,
                pop_item_ids=self._pop_item_ids,
                pop_weights=self._pop_weights,
                item2cat=self._item2cat,
                sampler_seed=self.sampler_seed,
            )

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
