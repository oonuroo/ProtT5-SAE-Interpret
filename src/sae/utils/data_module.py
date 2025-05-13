import os
import sys
sys.path.append("/home/onur/Desktop/Project/proj/src")

import multiprocessing
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils.utils import train_val_test_split


# data_module : 
    # 1. Loads embeddings from .h5 files
    # Splits proteins id`s train/val/test
    # Feed them into pytorch library 
    # Uses batch size 1, feeding sequences one by one since the tensor sizes are different 

class H5EmbeddingDataset(Dataset):
    ###  PyTorch Dataset for loading variable-length protein embeddings from an HDF5 file.
    ### Each embedding is shaped [L, 1024], where L is the number of residues.
    
    def __init__(self, h5_file_path, protein_ids):
        self.h5_file_path = h5_file_path
        self.protein_ids = protein_ids

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        prot_id = self.protein_ids[idx]
        with h5py.File(self.h5_file_path, "r") as h5f:
            embedding = torch.tensor(h5f[prot_id][:], dtype=torch.float32)
        return {"id": prot_id, "embedding": embedding}


# Data Module
class H5DataModule(pl.LightningDataModule):
    def __init__(self, h5_path, batch_size, num_workers=None,limit=None):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count() - 1
        self.limit = limit

    def setup(self, stage=None):
        with h5py.File(self.h5_path, "r") as h5f:
            self.all_ids = list(h5f.keys())
            if self.limit is not None:
                self.all_ids = self.all_ids[:self.limit]


        self.train_ids, self.val_ids, self.test_ids = train_val_test_split(self.all_ids)

    def train_dataloader(self):
        return DataLoader(
            H5EmbeddingDataset(self.h5_path, self.train_ids),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            H5EmbeddingDataset(self.h5_path, self.val_ids),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            H5EmbeddingDataset(self.h5_path, self.test_ids),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    def merge_train_val(self):         ## Merge train and val IDs for training without touching test set.
        self.train_ids = self.train_ids + self.val_ids
        self.val_ids = [] 
