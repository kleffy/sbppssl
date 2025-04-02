import numpy as np
import torch
from torch.utils.data import Dataset


import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SOCPatchDataset(Dataset):
    """
    Dataset for loading SOC patches extracted by EnMAPSOCPatchExtractor
    """
    def __init__(self, patches_dir, metadata_file, transform=None):
        """
        Args:
            patches_dir: Directory containing the .npy patch files
            metadata_file: Path to the extraction_metadata.json file
            transform: Optional transform to apply to the patches
        """
        self.patches_dir = Path(patches_dir)
        self.transform = transform

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Extract patch information
        self.patches = metadata['patches']

        # Verify all patch files exist
        valid_patches = []
        for patch_info in self.patches:
            patch_file = Path(patch_info['file_path'])
            if patch_file.exists():
                valid_patches.append(patch_info)
            else:
                logger.warning(f"Patch file not found: {patch_file}")

        self.patches = valid_patches
        logger.info(f"Loaded {len(self.patches)} valid SOC patches from metadata")

        # Extract SOC value statistics
        soc_values = [patch['soc_value'] for patch in self.patches]
        logger.info(f"SOC value range: {min(soc_values):.2f} - {max(soc_values):.2f}%, "
                   f"Mean: {np.mean(soc_values):.2f}%, Std: {np.std(soc_values):.2f}%")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]

        # Load patch data
        patch_file = patch_info['file_path']
        patch_data = np.load(patch_file)

        # Convert to torch tensor
        patch_tensor = torch.from_numpy(patch_data).float()

        # Get SOC value as target
        soc_value = patch_info['soc_value']
        soc_tensor = torch.tensor(soc_value, dtype=torch.float32)

        # Apply transforms if available
        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        return patch_tensor, soc_tensor