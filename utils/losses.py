import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, predictions, targets, mask=None):
        if self.loss_type == 'mse':
            if mask is not None:
                # Compute loss only on masked regions (where mask == 0)
                loss = F.mse_loss(predictions[mask == 0], targets[mask == 0])
            else:
                loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'l1':
            if mask is not None:
                loss = F.l1_loss(predictions[mask == 0], targets[mask == 0])
            else:
                loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'smooth_l1':
            if mask is not None:
                loss = F.smooth_l1_loss(predictions[mask == 0], targets[mask == 0])
            else:
                loss = F.smooth_l1_loss(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return loss


class PermutationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape [B, factorial(num_segments)] containing logits
            targets: Tensor of shape [B] containing permutation indices
        """
        return F.cross_entropy(predictions, targets)


class CombinedSSLLoss(nn.Module):
    def __init__(self, lambda_perm=0.5, rec_loss_type='mse', rec_loss_scale=1e-5):
        super().__init__()
        self.lambda_perm = lambda_perm
        self.rec_loss = ReconstructionLoss(rec_loss_type)
        self.perm_loss = PermutationLoss()
        self.rec_loss_scale = rec_loss_scale  # Scale factor to balance reconstruction loss
    
    def forward(self, predictions, targets, mask=None, original_x=None):
        """
        Combined loss for masked reconstruction and permutation prediction.
        
        Args:
            predictions: Dict containing 'reconstruction' and 'permutation' outputs
            targets: Dict containing permutation targets ('inverse_permutation')
            mask: Binary mask indicating masked positions (0 = masked)
            original_x: Original input tensor for reconstruction
        """
        losses = {}
        
        # Reconstruction loss
        if 'reconstruction' in predictions and original_x is not None:
            rec_loss = self.rec_loss(predictions['reconstruction'], original_x, mask)
            # Apply scaling to prevent the reconstruction loss from dominating
            scale = self.rec_loss_scale.item() if torch.is_tensor(self.rec_loss_scale) else float(self.rec_loss_scale)
            rec_loss = rec_loss * scale
            losses['rec_loss'] = rec_loss
        else:
            rec_loss = 0
        
        # Permutation prediction loss
        if 'permutation' in predictions and 'inverse_permutation' in targets:
            perm_loss = self.perm_loss(predictions['permutation'], targets['inverse_permutation'])
            losses['perm_loss'] = perm_loss
        else:
            perm_loss = 0
        
        # Combined loss
        total_loss = rec_loss + self.lambda_perm * perm_loss
        losses['total'] = total_loss
        
        return total_loss, losses


class SOCRegressionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, predictions, targets):
        """
        Calculate loss for SOC regression.
        
        Args:
            predictions: Tensor of predicted SOC values
            targets: Tensor of target SOC values
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return loss