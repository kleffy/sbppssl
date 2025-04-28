import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path
import random
import json
import logging
from typing import List, Tuple, Dict, Optional, Union, Callable
from sklearn.model_selection import train_test_split

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

logger = logging.getLogger(__name__)

class EnMAPDataset(Dataset):
    """
    Base class for loading EnMAP hyperspectral data and creating patches.
    """
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        patch_size: int = 32,
        patch_overlap: float = 0.1,
        sampling_strategy: str = 'random',
        max_patches_per_image: int = 50,
        min_valid_ratio: float = 0.7,
        bands: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        use_masks: bool = True,
        mask_suffix: str = '_mask.TIF',
        test_size: float = 0.2,
        val_size: float = 0.2,
        test_dir: Optional[str] = None,
        random_seed: int = 42,
        cache_data: bool = False,
        nodata_value: int = -32768,
        normalize: bool = True,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.sampling_strategy = sampling_strategy
        self.max_patches_per_image = max_patches_per_image
        self.min_valid_ratio = min_valid_ratio
        self.transform = transform
        self.augmentation = augmentation
        self.use_masks = use_masks
        self.mask_suffix = mask_suffix
        self.cache_data = cache_data
        self.nodata_value = nodata_value
        self.normalize = normalize
        self.random_seed = random_seed
        self.test_dir = Path(test_dir) if test_dir else None
        self.verbose = verbose
        
        # Important: Store bands attribute before using it
        self.bands = bands
        
        # Seed random number generator for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Find all EnMAP files based on mode
        if mode == 'test' and self.test_dir:
            # Use the separate test directory for test mode
            self.data_dir = self.test_dir
        
        self.file_list = self._get_files()
        if len(self.file_list) == 0:
            raise ValueError(f"No EnMAP data files found in {self.data_dir}")
        
        # Only log if verbose
        if self.verbose:
            logger.info(f"Found {len(self.file_list)} EnMAP data files")
        
        # Load first file to get image metadata
        with rasterio.open(self.file_list[0]) as src:
            self.num_bands = src.count
            if self.bands is None:
                self.bands = list(range(self.num_bands))
            else:
                # Ensure bands are in valid range
                if self.bands >= self.num_bands or self.bands < 0:
                    raise ValueError(f"Band indices must be between 0 and {self.num_bands-1}")
        
        # Only log if verbose
        if self.verbose:
            logger.info(f"Using {len(self.bands)} bands out of {self.num_bands}")
        
        # Initialize cache if needed
        self.data_cache = {} if self.cache_data else None
        
        # Create patches for data access
        self.patches = self._create_patches()
        
        # Split data for training/validation/test modes
        if mode in ['train', 'val', 'test']:
            self._split_data(test_size, val_size)
    
    def _get_files(self) -> List[Path]:
        """Get all EnMAP data files in the data directory."""
        files = []
        for ext in ['.TIF', '.TIFF', '.tif', '.tiff']:  
            files.extend([f for f in self.data_dir.glob(f"*{ext}") 
                         if not f.name.endswith(self.mask_suffix)])
        return sorted(files)
    
    def _create_patches(self) -> List[Dict]:
        """Create list of patches from data files using the specified sampling strategy."""
        patches = []
        
        for file_idx, file_path in enumerate(self.file_list):
            # Get file dimensions
            with rasterio.open(file_path) as src:
                height, width = src.height, src.width
            
            # Check if mask exists
            mask_path = file_path.with_name(file_path.stem + self.mask_suffix)
            has_mask = mask_path.exists() and self.use_masks
            
            # Load mask if available and we're filtering based on valid pixel ratio
            mask = None
            if has_mask and self.min_valid_ratio > 0:
                with rasterio.open(mask_path) as src:
                    mask = src.read(1) > 0  # Convert to boolean
            
            if self.sampling_strategy == 'stride':
                # Use stride-based patch sampling
                patches.extend(self._create_stride_patches(
                    file_idx, file_path, mask_path, has_mask, height, width, mask))
            else:
                # Use random patch sampling
                patches.extend(self._create_random_patches(
                    file_idx, file_path, mask_path, has_mask, height, width, mask))
        
        if self.verbose:
            logger.info(f"Created {len(patches)} patches of size {self.patch_size} from {len(self.file_list)} files using '{self.sampling_strategy}' sampling")
        return patches
    
    def _create_stride_patches(self, file_idx, file_path, mask_path, has_mask, height, width, mask) -> List[Dict]:
        """Create patches using stride-based sampling."""
        patches = []
        
        # Calculate stride based on overlap
        stride = int(self.patch_size * (1 - self.patch_overlap))
        stride = max(1, stride)  # Ensure stride is at least 1
        
        # Calculate number of patches in each dimension
        num_y = (height - self.patch_size) // stride + 1
        num_x = (width - self.patch_size) // stride + 1
        
        # Create patch entries
        for y_idx in range(num_y):
            y_start = y_idx * stride
            for x_idx in range(num_x):
                x_start = x_idx * stride
                
                # Ensure we don't go beyond image boundaries
                if y_start + self.patch_size <= height and x_start + self.patch_size <= width:
                    # Check if patch meets the minimum valid pixel ratio
                    if mask is not None and self.min_valid_ratio > 0:
                        patch_mask = mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                        valid_ratio = np.mean(patch_mask)
                        if valid_ratio < self.min_valid_ratio:
                            continue  # Skip patches with too few valid pixels
                    
                    patches.append({
                        'file_idx': file_idx,
                        'file_path': file_path,
                        'mask_path': mask_path if has_mask else None,
                        'x_start': x_start,
                        'y_start': y_start,
                        'has_mask': has_mask
                    })
        
        return patches
    
    def _create_random_patches(self, file_idx, file_path, mask_path, has_mask, height, width, mask) -> List[Dict]:
        """Create patches using random sampling."""
        patches = []
        
        # Determine maximum possible patch positions
        max_y = height - self.patch_size
        max_x = width - self.patch_size
        
        if max_y <= 0 or max_x <= 0:
            logging.warning(f"Image {file_path} is too small for patch size {self.patch_size}, skipping")
            return patches
        
        # Attempt to create up to max_patches_per_image valid patches
        attempts = 0
        max_attempts = self.max_patches_per_image * 3  # Allow for some failures
        
        while len(patches) < self.max_patches_per_image and attempts < max_attempts:
            attempts += 1
            
            # Generate random patch position
            y_start = random.randint(0, max_y)
            x_start = random.randint(0, max_x)
            
            # Check if patch meets the minimum valid pixel ratio
            if mask is not None and self.min_valid_ratio > 0:
                patch_mask = mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                valid_ratio = np.mean(patch_mask)
                if valid_ratio < self.min_valid_ratio:
                    continue  # Skip patches with too few valid pixels
            
            # Add patch to the list
            patches.append({
                'file_idx': file_idx,
                'file_path': file_path,
                'mask_path': mask_path if has_mask else None,
                'x_start': x_start,
                'y_start': y_start,
                'has_mask': has_mask
            })
        
        if attempts >= max_attempts and len(patches) < self.max_patches_per_image:
            logging.warning(f"Could only create {len(patches)} valid patches for {file_path} after {attempts} attempts")
        
        return patches
    
    def _split_data(self, test_size: float, val_size: float) -> None:
        """Split patches into training, validation, and test sets."""
        # Get unique file indices to split by file rather than patch
        file_indices = list(set(patch['file_idx'] for patch in self.patches))
        
        # Split file indices
        train_val_indices, test_indices = train_test_split(
            file_indices, test_size=test_size, random_state=self.random_seed
        )
        
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=self.random_seed
        )
        
        # Create train/val/test masks for patches
        self.train_mask = np.array([
            patch['file_idx'] in train_indices for patch in self.patches
        ])
        self.val_mask = np.array([
            patch['file_idx'] in val_indices for patch in self.patches
        ])
        self.test_mask = np.array([
            patch['file_idx'] in test_indices for patch in self.patches
        ])
        
        # Only log if verbose
        if self.verbose:
            logger.info(f"Split data: {np.sum(self.train_mask)} training patches, "
                       f"{np.sum(self.val_mask)} validation patches, "
                       f"{np.sum(self.test_mask)} test patches")
    
    def _load_patch(self, patch_info: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load a specific patch from a file."""
        file_path = patch_info['file_path']
        x_start = patch_info['x_start']
        y_start = patch_info['y_start']
        
        # Check if file is cached
        if self.cache_data and file_path in self.data_cache:
            data = self.data_cache[file_path]['data']
            mask = self.data_cache[file_path]['mask']
        else:
            # Read data
            with rasterio.open(file_path) as src:
                data = src.read(
                    indexes=[i+1 for i in self.bands],
                    window=rasterio.windows.Window(
                        x_start, y_start, self.patch_size, self.patch_size
                    )
                )
            
            # Read mask if available
            mask = None
            if patch_info['has_mask']:
                with rasterio.open(patch_info['mask_path']) as src:
                    mask = src.read(
                        1,
                        window=rasterio.windows.Window(
                            x_start, y_start, self.patch_size, self.patch_size
                        )
                    )
                    mask = mask > 0  # Convert to boolean
            
            # Cache if needed
            if self.cache_data:
                self.data_cache[file_path] = {
                    'data': data,
                    'mask': mask
                }
        
        return data, mask
    
    def __len__(self) -> int:
        """Return number of patches based on the mode."""
        if self.mode == 'train':
            return np.sum(self.train_mask)
        elif self.mode == 'val':
            return np.sum(self.val_mask)
        elif self.mode == 'test':
            return np.sum(self.test_mask)
        else:
            return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a data item by index."""
        # Map index to correct patch depending on mode
        if self.mode == 'train':
            # Get idx-th patch from training set
            actual_idx = np.where(self.train_mask)[0][idx]
        elif self.mode == 'val':
            # Get idx-th patch from validation set
            actual_idx = np.where(self.val_mask)[0][idx]
        elif self.mode == 'test':
            # Get idx-th patch from test set
            actual_idx = np.where(self.test_mask)[0][idx]
        else:
            actual_idx = idx
        
        patch_info = self.patches[actual_idx]
        
        # Load patch
        data, mask = self._load_patch(patch_info)
        
        # Apply mask if available
        if mask is not None:
            # Expand mask to match data dimensions
            mask_expanded = np.expand_dims(mask, axis=0).repeat(data.shape[0], axis=0)
            
            # Apply mask
            data = np.where(mask_expanded, data, self.nodata_value)
        
        # Handle no-data values and normalize
        if self.normalize:
            # Replace no-data with NaN for normalization
            data = data.astype(np.float32)
            data[data == self.nodata_value] = np.nan
            
            # Normalize each band independently
            for i in range(data.shape[0]):
                band_data = data[i]
                if not np.all(np.isnan(band_data)):
                    valid_data = band_data[~np.isnan(band_data)]
                    mean, std = np.mean(valid_data), np.std(valid_data)
                    if std > 0:
                        band_data = (band_data - mean) / std
                        data[i] = band_data
            
            # Replace NaN with zeros after normalization
            data = np.nan_to_num(data, nan=0.0)
        
        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)
        
        # Apply augmentation if in training mode
        if self.mode == 'train' and self.augmentation is not None:
            data = self.augmentation(data)
        
        # Ensure the data is contiguous
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        
        return {
            'data': torch.from_numpy(data).float(),
            'file_idx': patch_info['file_idx'],
            'coords': (patch_info['y_start'], patch_info['x_start']),
            'file_path': str(patch_info['file_path'])
        }
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )