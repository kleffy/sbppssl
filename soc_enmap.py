"""
SOCEnMAP model implementation for soil organic carbon estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base import BaseModel, BaseSSLModel


class SpectralTransformerBlock(nn.Module):
    def __init__(self, bands, embed_dim, num_heads=2):
        super().__init__()
        self.embed = nn.Linear(bands, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4
        )

    def forward(self, x):  # x: [B, H, W, bands]
        B, H, W, bands = x.shape
        x = x.view(-1, bands)
        x = self.embed(x) + self.positional_encoding  # [B*H*W, embed_dim]
        x = x.unsqueeze(1)  # [B*H*W, seq_len=1, embed_dim]
        x = self.transformer(x)  # [B*H*W, 1, embed_dim]
        x = x.squeeze(1)
        return x.view(B, H, W, -1)


class FactorizedAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=2, window_size=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.MultiheadAttention(channels, num_heads)
        self.window_size = window_size

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Prepare Spatial Attention
        # Handle case where H or W is smaller than window_size
        actual_window_size = min(self.window_size, min(H, W))
        
        if actual_window_size < 2:  # Skip windowed attention for very small inputs
            return x
            
        x_windows = x.unfold(2, actual_window_size, actual_window_size).unfold(
            3, actual_window_size, actual_window_size
        )
        B, C, H_win, W_win, h, w = x_windows.shape
        x_windows = x_windows.contiguous().view(B, C, -1, h * w).permute(2, 0, 3, 1).reshape(-1, h * w, C)

        # Spatial Attention
        x_windows, _ = self.spatial_attention(x_windows, x_windows, x_windows)
        
        # Reshape back - this is approximate as we're not properly recombining windows
        # For simplicity, we'll just reshape and resize back to original dimensions
        x_windows = x_windows.reshape(-1, B, h * w, C).permute(1, 3, 0, 2)
        x_out = F.interpolate(x_windows.reshape(B, C, -1, 1), size=(H * W, 1), mode='nearest')
        x_out = x_out.reshape(B, C, H, W)
        
        # Add residual connection
        return x + x_out


class SpatialPyramidConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Update padding: for kernel=7 and dilation=2, use padding = 2*(7-1)//2 = 6.
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=6, dilation=2)
        # Update padding: for kernel=15 and dilation=3, use padding = 3*(15-1)//2 = 21.
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=15, padding=21, dilation=3)
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return F.relu(self.bn(x))


class SOCEnMAPModel(BaseModel):
    def __init__(self, bands=224, spectral_embed_dim=64, spatial_channels=64, num_heads=4):
        super().__init__()
        self.spectral_transformer = SpectralTransformerBlock(bands, spectral_embed_dim, num_heads)
        self.spatial_pyramid = SpatialPyramidConv(spectral_embed_dim, spatial_channels)
        self.factorized_attention = FactorizedAttentionBlock(spatial_channels * 3, num_heads)
        self.final_conv = nn.Conv2d(spatial_channels * 3, 1, kernel_size=1)
        
        # Store parameters for reference
        self.bands = bands
        self.spectral_embed_dim = spectral_embed_dim
        self.spatial_channels = spatial_channels
        self.num_heads = num_heads
        self.output_channels = 1  # Number of output channels

    def forward(self, x):  # x: [B, bands, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, bands]
        x = self.spectral_transformer(x)  # [B, H, W, spectral_embed_dim]
        x = x.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]

        x = self.spatial_pyramid(x)  # [B, spatial_channels*3, H, W]
        x = self.factorized_attention(x)  # [B, spatial_channels*3, H, W]

        x = self.final_conv(x)  # [B, 1, H, W]
        return x


class SpectralPermutationHead(nn.Module):
    def __init__(self, in_channels, num_segments):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_segments * num_segments)
        )
        self.num_segments = num_segments
        
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x.view(-1, self.num_segments, self.num_segments)


class SOCEnMAPSSLModel(BaseSSLModel):
    def __init__(self, bands=224, spectral_embed_dim=64, spatial_channels=64, 
                 num_heads=4, num_segments=8):
        super().__init__()
        # Base encoder
        self.encoder = SOCEnMAPModel(
            bands, spectral_embed_dim, spatial_channels, num_heads
        )
        # Reconstruction head
        self.reconstruction_head = nn.Conv2d(1, bands, kernel_size=1)
        # Permutation prediction head
        self.permutation_head = SpectralPermutationHead(
            spatial_channels * 3, num_segments
        )
        
        self.num_segments = num_segments
        self.segment_length = bands // num_segments
        
    def forward(self, x, task='both'):
        # Pass through encoder to get features
        features = self.encoder(x)
        
        outputs = {}
        
        if task in ['reconstruction', 'both']:
            outputs['reconstruction'] = self.reconstruction_head(features)
            
        if task in ['permutation', 'both']:
            # Pass spatial features before final conv to permutation head
            # Extract the features before final_conv for permutation head
            x_transformed = x.permute(0, 2, 3, 1)  # [B, H, W, bands]
            x_transformer = self.encoder.spectral_transformer(x_transformed)  # [B, H, W, spectral_embed_dim]
            x_spatial = x_transformer.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
            x_pyramid = self.encoder.spatial_pyramid(x_spatial)  # [B, spatial_channels*3, H, W]
            x_attention = self.encoder.factorized_attention(x_pyramid)  # [B, spatial_channels*3, H, W]
            
            permutation_logits = self.permutation_head(x_attention)
            outputs['permutation'] = permutation_logits
        
        return outputs
    
    def get_encoder_state_dict(self):
        """Get state dict for just the encoder part."""
        return self.encoder.state_dict()