"""
Base model classes with common interfaces for all SOC estimation models.
"""
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all SOC estimation models."""
    
    def forward(self, x):
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError
    
    def mask_pooling(self, predictions, masks):
        """Pool predictions using field masks"""
        masked_preds = predictions * masks
        return masked_preds.sum(dim=(2, 3)) / (masks.sum(dim=(2, 3)) + 1e-8)
    
    def load_pretrained(self, state_dict):
        """
        Load pretrained weights with compatibility handling.
        Subclasses should override this if needed.
        
        Args:
            state_dict: State dict from pretrained model
        """
        # Filter prefix from state dict if needed
        filtered_dict = {}
        for k, v in state_dict.items():
            # Remove 'encoder.' prefix if present
            if k.startswith('encoder.'):
                filtered_dict[k[8:]] = v
            # Skip SSL-specific layers
            elif not k.startswith('reconstruction_head.') and not k.startswith('permutation_head.'):
                filtered_dict[k] = v
                
        # Load with flexible strict setting
        incompatible_keys = self.load_state_dict(filtered_dict, strict=False)
        
        # Return info about loading status
        return {
            'success': True,
            'incompatible_keys': incompatible_keys
        }


class BaseSSLModel(nn.Module):
    """Base class for all SSL models used in pretraining."""
    
    def __init__(self):
        super().__init__()
        self.encoder = None
        
    def forward(self, x, task='both'):
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError
    
    def get_encoder(self):
        """Get the encoder part of the SSL model."""
        return self.encoder
    
    def get_encoder_state_dict(self):
        """Get state dict for just the encoder part."""
        if self.encoder is None:
            raise ValueError("Encoder not initialized")
            
        return self.encoder.state_dict()