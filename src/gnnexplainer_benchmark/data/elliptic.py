from typing import Literal

import os

import pandas as pd
import torch
from torch_geometric.data import Data

class BCDataset:
    """
    Universal dataset wrapper for fraud datasets.
    Automatically loads features, edges, labels and masks.
        Currently only 'elliptic' is implemented.
    """
    def __init__(self,
                 type: str,
                 **kwargs):
        """
        Args:
            type (str): Dataset to load. Supported: 'elliptic', 'ibm'.
            **kwargs: Forwarded to dataset-specific loader.
        """
        self.type = type.lower()
        transforms = {
            'elliptic': self._load_elliptic,
        }
        if self.type not in transforms:
            raise ValueError(f"Unsupported dataset type: {self.type}. "
                             f"Supported types: {list(transforms.keys())}")
        transforms[self.type](**kwargs)


    def _load_elliptic(self,
                       path: str = "datasets/elliptic",
                       data_path: str = None,
                       classes: dict = {'unknown': 2, '1': 1, '2': 0},
                       directed: bool = False,
                       time_splits: list = [30, 40]):
        """
        Load the Elliptic dataset into:
          - self.features   (FloatTensor[N, F])
          - self.labels     (LongTensor[N])
          - self.edge_index (LongTensor[2, E])
          - self.train_mask (BoolTensor[N])
          - self.val_mask   (BoolTensor[N])
          - self.test_mask  (BoolTensor[N])
        """
        if data_path and os.path.exists(data_path):
            data = torch.load(data_path, map_location="cpu", weights_only=False)
            self.features = data.x
            self.labels = data.y
            self.edge_index = data.edge_index
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
            self.test_mask = data.test_mask
            self._features_are_model_ready = True
            if hasattr(data, "time_step"):
                self._time_step = data.time_step
            return

        feat_df = pd.read_csv(f"{path}/elliptic_txs_features.csv", header=None)
        edge_df = pd.read_csv(f"{path}/elliptic_txs_edgelist.csv")
        class_df = pd.read_csv(f"{path}/elliptic_txs_classes.csv")

        feat_df = feat_df.rename(columns={0: 'txId', 1: 'time_step'})
        feat_array = feat_df.loc[:, 'time_step':].values
        self.features = torch.tensor(feat_array, dtype=torch.float)
        if self.features.size()[1] == 0:
            self.features = torch.ones(self.features.shape[0], 1, dtype=torch.float)

        mapped = class_df['class'].map(classes)
        self.labels = torch.tensor(mapped.values, dtype=torch.int64)

        nodes = feat_df['txId']
        map_id = {j: i for i, j in enumerate(nodes)}

        edges_df = edge_df[['txId1', 'txId2']].copy()

        # Handle directionality
        if not directed:
            edges_rev = edges_df.rename(columns={'txId1': 'txId2', 'txId2': 'txId1'})
            edges_df = pd.concat([edges_df, edges_rev], ignore_index=True)

        edges_df['txId1'] = edges_df['txId1'].map(map_id)
        edges_df['txId2'] = edges_df['txId2'].map(map_id)

        # Drop invalid and redundant edges
        edges_df = edges_df.dropna().astype(int)
        edges_df = edges_df[edges_df['txId1'] != edges_df['txId2']]  # remove self-loops
        edges_df = edges_df.drop_duplicates().reset_index(drop=True)

        edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
        self.edge_index = edge_index

        time_step = torch.tensor(feat_df['time_step'].values, dtype=torch.long)
        self._time_step = time_step
        assert len(time_splits) == 2, "time_splits must have exactly two values"
        t0, t1 = time_splits

        known = (self.labels != classes.get('unknown', 2))
        self.train_mask = (time_step < t0) & known
        self.val_mask = ((time_step >= t0) & (time_step < t1)) & known
        self.test_mask = (time_step >= t1) & known

    def _build_val_from_train(
        self,
        seed: int = 42,
        val_ratio: float = 0.2,
        temporal_split: bool = True
    ) -> None:
        """
        Build validation mask from the training mask only (never from test).
        This is used as a safe fallback when no validation mask is provided.
        """
        train_mask = self.train_mask.clone().to(torch.bool)
        train_indices = torch.nonzero(train_mask, as_tuple=False).view(-1)
        if train_indices.numel() < 2:
            raise ValueError("Need at least 2 training samples to derive a validation split.")

        val_count = max(1, int(round(train_indices.numel() * float(val_ratio))))
        val_count = min(val_count, train_indices.numel() - 1)

        if temporal_split:
            if hasattr(self, "_time_step"):
                ordered = train_indices[torch.argsort(self._time_step[train_indices])]
            else:
                ordered = train_indices
            val_indices = ordered[-val_count:]
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            perm = torch.randperm(train_indices.numel(), generator=generator)
            val_indices = train_indices[perm[:val_count]]

        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        val_mask[val_indices] = True
        train_mask[val_indices] = False

        self.train_mask = train_mask
        self.val_mask = val_mask

    def get_masks(self, seed: int = 42, val_ratio: float = 0.2, temporal_split: bool = True):
        """
        Returns the train, validation, and test masks.
        """
        if not hasattr(self, 'train_mask') or not hasattr(self, 'test_mask'):
            raise AttributeError("Train/test masks are not defined. Initialize the dataset first.")
        if (not hasattr(self, 'val_mask')) or (self.val_mask is None) or (self.val_mask.sum().item() == 0):
            self._build_val_from_train(seed=seed, val_ratio=val_ratio, temporal_split=temporal_split)

        train_mask = self.train_mask.clone().to(torch.bool)
        val_mask = self.val_mask.clone().to(torch.bool)
        test_mask = self.test_mask.clone().to(torch.bool)

        if torch.logical_and(train_mask, val_mask).any().item():
            raise AssertionError("train_mask and val_mask must be disjoint.")
        if torch.logical_and(train_mask, test_mask).any().item():
            raise AssertionError("train_mask and test_mask must be disjoint.")
        if torch.logical_and(val_mask, test_mask).any().item():
            raise AssertionError("val_mask and test_mask must be disjoint.")

        return (
            train_mask,
            val_mask,
            test_mask
        )

    def to_torch_data(self) -> Data:
        """
        Convert stored tensors (features, labels, edge_index, masks) into
        a single torch_geometric.data.Data object.
        """
        should_drop_time_step = (
            self.type == "elliptic"
            and not getattr(self, "_features_are_model_ready", False)
            and self.features.size(1) > 1
        )
        x = self.features[:, 1:] if should_drop_time_step else self.features
        y = self.labels
        edge_index = self.edge_index

        data = Data(x=x, y=y, edge_index=edge_index)

        data.train_mask = self.train_mask.clone().to(torch.bool)
        data.val_mask   = self.val_mask.clone().to(torch.bool)
        data.test_mask  = self.test_mask.clone().to(torch.bool)
        if hasattr(self, "_time_step"):
            data.time_step = self._time_step.clone()

        return data
