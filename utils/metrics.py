import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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
    Calculate metrics for permutation prediction.
    
    Args:
        predictions: Permutation predictions [B, num_segments, num_segments]
        targets: Permutation targets [B, num_segments]
        
    Returns:
        Dictionary of metrics
    """
    # Convert to CPU
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    
    # Get predicted indices
    pred_indices = torch.argmax(predictions, dim=2)  # [B, num_segments]
    
    # Calculate segment accuracy (per-position accuracy)
    correct = (pred_indices == targets).float()
    segment_accuracy = torch.mean(correct).item()
    
    # Calculate permutation accuracy (exact match)
    permutation_accuracy = torch.mean(torch.all(correct, dim=1).float()).item()
    
    # Calculate per-position accuracy
    num_segments = targets.shape[1]
    position_accuracies = []
    
    for pos in range(num_segments):
        pos_accuracy = torch.mean((pred_indices[:, pos] == targets[:, pos]).float()).item()
        position_accuracies.append(pos_accuracy)
    
    # Calculate confusion matrix
    batch_size = targets.shape[0]
    confusion_matrix = torch.zeros(num_segments, num_segments, dtype=torch.int)
    
    for i in range(batch_size):
        for j in range(num_segments):
            target_pos = targets[i, j].item()
            pred_pos = pred_indices[i, j].item()
            confusion_matrix[target_pos, pred_pos] += 1
    
    return {
        'segment_accuracy': segment_accuracy,
        'permutation_accuracy': permutation_accuracy,
        'position_accuracies': position_accuracies,
        'confusion_matrix': confusion_matrix
    }