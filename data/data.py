import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
from pathlib import Path
import random
import logging
from sklearn.model_selection import train_test_split
import json

from .base_dataset import EnMAPDataset
from .soc_patch_dataset import SOCPatchDataset


class EnMAPDataModule:
    def __init__(self, config):
        self.config = config
        self.unlabeled_data_dir = config.get('unlabeled_data_dir', './data/unlabeled')
        self.labeled_patches_dir = config.get('labeled_patches_dir', './data/labeled')
        self.metadata_file = config.get('metadata_file', './data/metadata.json')
        self.batch_size = config.get('batch_size', 16)
        self.val_batch_size = config.get('val_batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.patch_size = config.get('patch_size', 64)
        self.bands = config.get('bands', None)
        self.val_size = config.get('val_size', 0.2)
        self.seed = config.get('seed', 42)
    
    def get_pretraining_loader(self):
        dataset = EnMAPDataset(
            data_dir=self.unlabeled_data_dir,
            mode='train',
            patch_size=self.patch_size,
            bands=self.bands
        )
        
        # Split into train and validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return {'train': train_loader, 'val': val_loader}
    
    def get_finetuning_loaders(self):
        dataset = SOCPatchDataset(
            patches_dir=self.labeled_patches_dir,
            metadata_file=self.metadata_file
        )
        
        # Split into train and validation
        val_size = int(self.val_size * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader