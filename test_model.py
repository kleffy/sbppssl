from models import create_model
import torch

def main():
    try:
        # Create dummy input: batch size 1, 224 channels, height and width of 64
        dummy_input = torch.randn(1, 224, 64, 64)
        
        # Test creating the main model
        model_main = create_model('socenmap', is_ssl=False, bands=224, spectral_embed_dim=64, spatial_channels=64, num_heads=4)
        # print("Main model created:", model_main)
        output_main = model_main(dummy_input)
        print("Main model output shape:", output_main.shape)
        
        # Test creating the SSL model
        model_ssl = create_model('socenmap', is_ssl=True, bands=224, spectral_embed_dim=64, spatial_channels=64, num_heads=4)
        # print("SSL model created:", model_ssl)
        output_ssl = model_ssl(dummy_input)
        print("SSL model output:")
        print("  Reconstruction output shape:", output_ssl['reconstruction'].shape)
        print("  Permutation output shape:", output_ssl['permutation'].shape)
        
    except Exception as e:
        print("Error during model creation:", e)

if __name__ == "__main__":
    main()
