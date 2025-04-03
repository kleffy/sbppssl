import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import itertools
import math


def calculate_reconstruction_metrics(original, reconstructed, mask=None):
    """
    Calculate metrics for reconstruction quality.
    
    Args:
        original: Original input tensor [B, C, H, W]
        reconstructed: Reconstructed output tensor [B, C, H, W]
        mask: Optional mask tensor [B, C, H, W] (0 = masked regions)
        
    Returns:
        Dictionary of metrics
    """
    # Move to CPU and convert to numpy
    original = original.detach().cpu().numpy()
    reconstructed = reconstructed.detach().cpu().numpy()
    
    if mask is not None:
        mask = mask.detach().cpu().numpy()
        
    # Calculate metrics only on masked regions if mask is provided
    if mask is not None:
        # Inverse mask to focus on masked regions (0 -> 1, 1 -> 0)
        inv_mask = 1.0 - mask
        
        # Apply mask
        masked_original = original * inv_mask
        masked_reconstructed = reconstructed * inv_mask
        
        # Focus on non-zero entries
        non_zero = inv_mask > 0
        if np.sum(non_zero) > 0:
            original_flat = masked_original[non_zero]
            reconstructed_flat = masked_reconstructed[non_zero]
        else:
            # If no masked regions, use all pixels
            original_flat = original.flatten()
            reconstructed_flat = reconstructed.flatten()
    else:
        original_flat = original.flatten()
        reconstructed_flat = reconstructed.flatten()
    
    # Calculate MSE
    mse = np.mean((original_flat - reconstructed_flat) ** 2)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    # Calculate MAE
    mae = np.mean(np.abs(original_flat - reconstructed_flat))
    
    # Calculate PSNR
    # Avoid division by zero
    if mse > 0:
        max_val = max(np.max(original_flat), np.max(reconstructed_flat))
        psnr_val = 20 * np.log10(max_val / np.sqrt(mse)) if max_val > 0 else 0
    else:
        psnr_val = 100.0  # Perfect reconstruction
    
    # Calculate SSIM for each sample and average
    ssim_vals = []
    batch_size = original.shape[0]
    
    # Use middle band for SSIM (2D image)
    middle_band = original.shape[1] // 2
    
    for i in range(batch_size):
        orig_img = original[i, middle_band]
        recon_img = reconstructed[i, middle_band]
        
        # Normalize images for SSIM
        orig_img = (orig_img - np.min(orig_img)) / (np.max(orig_img) - np.min(orig_img) + 1e-8)
        recon_img = (recon_img - np.min(recon_img)) / (np.max(recon_img) - np.min(recon_img) + 1e-8)
        
        ssim_val = ssim(orig_img, recon_img, data_range=1.0)
        ssim_vals.append(ssim_val)
    
    avg_ssim = np.mean(ssim_vals)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'psnr': psnr_val,
        'ssim': avg_ssim
    }


def calculate_permutation_metrics(predictions, targets):
    """
    Calculate metrics for segment-wise permutation prediction.
    
    Args:
        predictions: Permutation predictions [B, factorial(num_segments)]
        targets: Permutation target indices [B]
        
    Returns:
        Dictionary of metrics
    """
    
    # Convert to CPU
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    
    # Get predicted permutation indices
    pred_indices = torch.argmax(predictions, dim=1)  # [B]
    
    # Calculate top-level accuracy (exact permutation match)
    permutation_accuracy = (pred_indices == targets).float().mean().item()
    
    # Get batch size and infer number of segments from predictions shape
    batch_size = predictions.shape[0]
    num_permutations = predictions.shape[1]
    
    # Calculate num_segments from factorial
    # Solve for n where n! = num_permutations
    num_segments = 1
    while math.factorial(num_segments) < num_permutations:
        num_segments += 1
    
    # Generate all possible permutations
    all_permutations = list(itertools.permutations(range(num_segments)))
    
    # Convert permutation indices to actual permutations
    segment_preds = []
    segment_targets = []
    
    for i in range(batch_size):
        pred_perm_idx = pred_indices[i].item()
        target_perm_idx = targets[i].item()
        
        if pred_perm_idx < len(all_permutations):
            pred_perm = all_permutations[pred_perm_idx]
            segment_preds.append(pred_perm)
        else:
            # Handle out-of-range indices
            pred_perm = tuple(range(num_segments))
            segment_preds.append(pred_perm)
            
        if target_perm_idx < len(all_permutations):
            target_perm = all_permutations[target_perm_idx]
            segment_targets.append(target_perm)
        else:
            # Handle out-of-range indices
            target_perm = tuple(range(num_segments))
            segment_targets.append(target_perm)
    
    # Calculate per-segment accuracy
    segment_accuracies = []
    for seg_idx in range(num_segments):
        correct_count = 0
        for i in range(batch_size):
            if segment_preds[i][seg_idx] == segment_targets[i][seg_idx]:
                correct_count += 1
        segment_accuracies.append(correct_count / batch_size)
    
    # Calculate segment-wise accuracy (average across all segments)
    segment_accuracy = sum(segment_accuracies) / len(segment_accuracies)
    
    # Create a confusion matrix for segment positions
    confusion_matrix = torch.zeros(num_segments, num_segments, dtype=torch.int)
    for i in range(batch_size):
        for seg_idx in range(num_segments):
            target_pos = segment_targets[i][seg_idx]
            pred_pos = segment_preds[i][seg_idx]
            confusion_matrix[target_pos, pred_pos] += 1
    
    return {
        'permutation_accuracy': permutation_accuracy,  # Exact permutation match
        'segment_accuracy': segment_accuracy,          # Average segment position accuracy
        'segment_accuracies': segment_accuracies,      # Per-segment position accuracy
        'confusion_matrix': confusion_matrix           # Position confusion matrix
    }