import torch
import numpy as np
import math

class CurriculumScheduler:
    """
    Scheduler for curriculum learning difficulty.
    """
    def __init__(
        self,
        dataloader,
        initial_difficulty: float = 0.2,
        final_difficulty: float = 1.0,
        epochs_to_max_difficulty: int = 50
    ):
        self.dataloader = dataloader
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.epochs_to_max_difficulty = max(1, epochs_to_max_difficulty)
    
    def step(self, epoch: int):
        """Update difficulty based on current epoch"""
        if not hasattr(self.dataloader.dataset, 'permutation_difficulty'):
            return  # Dataset doesn't support curriculum learning
        
        if epoch >= self.epochs_to_max_difficulty:
            new_difficulty = self.final_difficulty
        else:
            # Linear scaling of difficulty
            progress = epoch / self.epochs_to_max_difficulty
            new_difficulty = self.initial_difficulty + progress * (self.final_difficulty - self.initial_difficulty)
        
        # Update dataset difficulty
        self.dataloader.dataset.permutation_difficulty = new_difficulty
        return new_difficulty

def apply_spectral_spatial_mask(x, mask_ratio=0.6, block_size=8):
    """Apply both spectral and spatial masking to the input tensor."""
    B, bands, H, W = x.shape
    mask = torch.ones_like(x)

    # Spatial Masking (random block-wise)
    for b in range(B):
        for _ in range(int(mask_ratio * (H * W) / (block_size ** 2))):
            # Ensure block doesn't go out of bounds
            top = torch.randint(0, max(1, H - block_size + 1), (1,))
            left = torch.randint(0, max(1, W - block_size + 1), (1,))
            
            # Handle small images where block_size > H or W
            actual_block_size_h = min(block_size, H - top)
            actual_block_size_w = min(block_size, W - left)
            
            mask[b, :, top:top + actual_block_size_h, left:left + actual_block_size_w] = 0

    # Spectral Masking (band-wise)
    num_bands_to_mask = int(bands * mask_ratio)
    bands_to_mask = torch.randperm(bands)[:num_bands_to_mask]
    mask[:, bands_to_mask, :, :] = 0

    x_masked = x * mask
    return x_masked, mask


def segment_and_permute_spectra(x, num_segments=8, return_targets=True):
    """
    Segment and permute the spectral dimension of the input tensor with improved handling
    of uneven divisions. Generates both sequence-based and index-based permutation targets.
    
    Args:
        x: Tensor of shape [B, C, H, W] where C is the spectral dimension
        num_segments: Number of segments to divide the spectral dimension into
        return_targets: Whether to return permutation targets for training
        
    Returns:
        permuted_x: Tensor with permuted spectral segments
        targets: Dict containing permutation indices and inverse permutation (if return_targets=True)
    """
    B, C, H, W = x.shape
    
    # Calculate base segment length and remainder
    base_segment_length = C // num_segments
    remainder = C % num_segments
    
    # Create permutation for each batch item
    permutation_targets = []
    permutation_indices = []  # New: store single indices for each permutation
    permuted_x = torch.zeros_like(x)
    
    for b in range(B):
        # Generate random permutation
        permutation = torch.randperm(num_segments)
        
        # Calculate segment boundaries and apply permutation
        cur_idx = 0  # Current index in the permuted tensor
        
        for i in range(num_segments):
            # Get permuted index
            p_idx = permutation[i].item()
            
            # Calculate segment length for this segment
            seg_length = base_segment_length + (1 if p_idx < remainder else 0)
            
            # Calculate original start and end indices
            orig_start = sum([base_segment_length + (1 if j < remainder else 0) for j in range(p_idx)])
            orig_end = orig_start + seg_length
            
            # Copy to new position
            permuted_x[b, cur_idx:cur_idx+seg_length, :, :] = x[b, orig_start:orig_end, :, :]
            cur_idx += seg_length
        
        if return_targets:
            # Calculate inverse permutation (to go from shuffled to original)
            inverse_permutation = torch.zeros_like(permutation)
            for i, p in enumerate(permutation):
                inverse_permutation[p] = i
            
            permutation_targets.append(inverse_permutation)
            
            # New: Convert permutation to a single index using Factorial Number System
            perm_seq = permutation.tolist()
            
            # Step 1: Compute Lehmer code
            lehmer = []
            elements = list(range(num_segments))
            for i, val in enumerate(perm_seq):
                idx = elements.index(val)
                lehmer.append(idx)
                elements.pop(idx)
            
            # Step 2: Convert Lehmer code to a single index
            index = 0
            for i, val in enumerate(lehmer):
                index += val * math.factorial(num_segments - 1 - i)
            
            permutation_indices.append(index)
    
    if return_targets:
        targets = {
            'permutation': torch.stack(permutation_targets),
            # Use the sequence-based inverse permutation (for compatibility/debugging)
            'inverse_permutation_seq': torch.stack(permutation_targets),
            # New: Add the single-index version (for cross_entropy loss)
            'inverse_permutation': torch.tensor(permutation_indices, device=x.device, dtype=torch.long)
        }
        return permuted_x, targets
    
    return permuted_x


def calculate_metrics(predictions, targets):
    """Calculate regression metrics for SOC prediction."""
    # Ensure inputs are numpy arrays for consistent calculation
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Coefficient of Determination (R²)
    ss_total = np.sum((targets - np.mean(targets)) ** 2)
    ss_residual = np.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Ratio of Performance to Deviation (RPD)
    rpd = np.std(targets) / rmse if rmse > 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rpd': rpd
    }


def get_curriculum_segments(epoch, max_epochs, min_segments=3, max_segments=8):
    """Determine the number of segments based on training progress for curriculum learning."""
    # Linear progression from min to max segments
    progress = min(1.0, epoch / (max_epochs * 0.7))  # Reach max at 70% of training
    segments_float = min_segments + progress * (max_segments - min_segments)
    num_segments = int(segments_float)
    
    return num_segments