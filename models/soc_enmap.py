import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralTransformerBlock(nn.Module):
    def __init__(self, bands, embed_dim, num_heads=2):
        super().__init__()
        self.bands = bands
        self.embed_dim = embed_dim
        
        # Use a simple linear layer for embedding
        self.embedding = nn.Linear(bands, embed_dim)
        
        # Create positional encoding
        self.register_buffer(
            "positional_encoding", 
            torch.randn(1, 1, embed_dim)
        )
        
        # Use a simple MLP instead of transformer for memory efficiency
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Add layer norm for stability
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: [B, H, W, bands]
        B, H, W, bands = x.shape
        
        # Reshape to [B*H*W, bands]
        x_flat = x.reshape(-1, bands)
        
        # Simple linear embedding
        x_embed = self.embedding(x_flat)  # [B*H*W, embed_dim]
        
        # Add positional encoding
        x_embed = x_embed + self.positional_encoding
        
        # Apply MLP with residual connection
        x_embed = x_embed + self.mlp(self.layer_norm(x_embed))
        
        # Reshape back to [B, H, W, embed_dim]
        return x_embed.view(B, H, W, -1)


class SimpleAttentionBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, H, W]
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x


class SpatialPyramidConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use a simpler implementation with smaller kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return F.relu(self.bn(x))


class SOCEnMAPModel(nn.Module):
    def __init__(self, bands=224, spectral_embed_dim=32, spatial_channels=16, num_heads=2):
        super().__init__()
        # Further reduced dimensions for memory efficiency
        self.spectral_transformer = SpectralTransformerBlock(bands, spectral_embed_dim, num_heads)
        self.spatial_pyramid = SpatialPyramidConv(spectral_embed_dim, spatial_channels)
        self.attention = SimpleAttentionBlock(spatial_channels * 3)
        self.final_conv = nn.Conv2d(spatial_channels * 3, 1, kernel_size=1)

    def forward(self, x):  # x: [B, bands, H, W]
        # Permute to [B, H, W, bands] for spectral transformer
        x = x.permute(0, 2, 3, 1)
        x = self.spectral_transformer(x)
        
        # Permute back to [B, embed_dim, H, W] for convolutional layers
        x = x.permute(0, 3, 1, 2)
        
        # Apply spatial pyramid and attention
        x = self.spatial_pyramid(x)
        x = self.attention(x)
        
        # Final 1x1 convolution to get output
        return self.final_conv(x)


class SpectralPermutationHead(nn.Module):
    def __init__(self, in_channels, num_segments):
        super().__init__()
        # Simpler implementation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, math.factorial(num_segments))

    def forward(self, x):
        x = self.gap(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class SOCEnMAPSSLModel(nn.Module):
    def __init__(self, bands=224, spectral_embed_dim=32, spatial_channels=16,
                 num_heads=2, num_segments=4):
        super().__init__()
        # Reduced dimensions for memory efficiency
        assert bands % num_segments == 0, "Bands must be divisible by num_segments"

        # Initialize encoder
        self.encoder = SOCEnMAPModel(bands, spectral_embed_dim, spatial_channels, num_heads)

        # Reconstruction head - simple 1x1 convolution
        self.reconstruction_head = nn.Conv2d(1, bands, kernel_size=1)

        # Permutation prediction head
        perm_head_in_channels = spatial_channels * 3
        self.permutation_head = SpectralPermutationHead(perm_head_in_channels, num_segments)

        # Store segmentation info
        self.num_segments = num_segments
        self.segment_length = bands // num_segments

    def forward(self, x, task='both'):
        outputs = {}

        # Shared encoder forward pass
        x_transformed = x.permute(0, 2, 3, 1)  # [B, H, W, bands]
        x_spectral = self.encoder.spectral_transformer(x_transformed)  # [B, H, W, embed_dim]
        x_spatial = x_spectral.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]

        # Spatial feature extraction
        x_pyramid = self.encoder.spatial_pyramid(x_spatial)  # [B, spatial_channels*3, H, W]
        x_attention = self.encoder.attention(x_pyramid)  # [B, spatial_channels*3, H, W]

        # Compute tasks based on what's requested
        if task in ['reconstruction', 'both']:
            recon_features = self.encoder.final_conv(x_attention)  # [B, 1, H, W]
            outputs['reconstruction'] = self.reconstruction_head(recon_features)  # [B, bands, H, W]

        if task in ['permutation', 'both']:
            permutation_logits = self.permutation_head(x_attention)  # [B, num_permutations]
            outputs['permutation'] = permutation_logits

        return outputs

    def get_encoder_state_dict(self):
        """Get state dict for just the encoder part."""
        return self.encoder.state_dict()


# Test code - with very small dimensions
if __name__ == "__main__":
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Creating model...")
    # Use an extremely lightweight model for testing
    model = SOCEnMAPSSLModel(
        bands=224, 
        spectral_embed_dim=16,  # Reduced from 32
        spatial_channels=8,     # Reduced from 16
        num_heads=2, 
        num_segments=8         # Reduced from 4 to make factorial smaller
    )
    
    print("Creating test input...")
    # Test with tiny spatial dimensions
    dummy_input = torch.randn(2, 224, 128, 128)  # Reduced from 32x32 to 16x16
    
    print("Running forward pass...")
    with torch.no_grad():  # Disable gradient tracking
        outputs = model(dummy_input, task='both')
    
    print('Reconstruction output shape:', outputs['reconstruction'].shape)
    print('Permutation output shape:', outputs['permutation'].shape)
    print("Test successful!")