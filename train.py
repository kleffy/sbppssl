#!/usr/bin/env python
import argparse
import yaml
import os
import torch
import logging
import numpy as np
import random
from datetime import datetime

from models import create_model
from data.data import EnMAPDataModule
from trainers.trainers import PretrainingTrainer, FinetuningTrainer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    config_path = '/vol/research/RobotFarming/Projects/sbppssl/config/config.yaml'
    parser = argparse.ArgumentParser(description="Training pipeline for SOC estimation models")
    parser.add_argument('--config', type=str, default=config_path, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_name = config.get('experiment_name', 'default_experiment')
    output_dir = os.path.join(config.get('output_dir', './output'), experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    config['output_dir'] = output_dir
    # Save the config file to the output directory
    config_file_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Setup
    setup_seed(config.get('seed', 42))
    setup_logging(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Get model configuration
    model_name = config.get('model_name', 'socenmap')
    logging.info(f"Using model: {model_name}")
    
    # Create data module
    data_module = EnMAPDataModule(config)
    
    # Run either pretraining or fine-tuning or both
    if config.get('run_pretraining', True):
        logging.info("Starting pretraining phase")
        
        # Create pretraining model using factory
        ssl_model = create_model(
            model_name, 
            is_ssl=True,
            bands=config.get('in_channels', 224),
            spectral_embed_dim=config.get('spectral_embed_dim', 64),
            spatial_channels=config.get('spatial_channels', 64),
            num_heads=config.get('num_heads', 4),
            num_segments=config.get('num_segments', 8)
        ).to(device)
        
        # Setup pretraining
        pretraining_trainer = PretrainingTrainer(ssl_model, config, device)
        
        # Run pretraining
        pretrained_path = pretraining_trainer.train(data_module.get_pretraining_loader())
        logging.info(f"Pretraining completed. Model saved at: {pretrained_path}")
        
        # Update config with pretrained model path
        config['pretrained_model_path'] = str(pretrained_path)
    
    if config.get('run_finetuning', True):
        logging.info("Starting fine-tuning phase")
        
        # Create base model first
        soc_model = create_model(
            model_name,
            is_ssl=False,
            bands=config.get('in_channels', 224),
            spectral_embed_dim=config.get('spectral_embed_dim', 64),
            spatial_channels=config.get('spatial_channels', 64),
            num_heads=config.get('num_heads', 4)
        ).to(device)
        
        # Get pretrained path
        pretrained_path = config.get('pretrained_model_path', None)
        
        if pretrained_path and os.path.exists(pretrained_path):
            logging.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            
            # Create a temporary instance of the SSL model to get encoder weights
            temp_ssl_model = create_model(
                model_name, 
                is_ssl=True,
                bands=config.get('in_channels', 224),
                spectral_embed_dim=config.get('spectral_embed_dim', 64),
                spatial_channels=config.get('spatial_channels', 64),
                num_heads=config.get('num_heads', 4),
                num_segments=config.get('num_segments', 8)
            ).to(device)
            
            # Load the full SSL model weights
            temp_ssl_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Use get_encoder_state_dict() to extract just the encoder weights
            encoder_state_dict = temp_ssl_model.get_encoder_state_dict()
            
            # Load these weights into the fine-tuning model
            soc_model.load_state_dict(encoder_state_dict, strict=False)
            
            logging.info("Pretrained encoder weights loaded successfully")
        else:
            logging.warning("No pretrained weights found, starting fine-tuning from scratch")
        
        # Setup fine-tuning
        finetuning_trainer = FinetuningTrainer(soc_model, config, device)
        
        # Run fine-tuning
        train_loader, val_loader = data_module.get_finetuning_loaders()
        finetuned_path = finetuning_trainer.train(train_loader, val_loader)
        logging.info(f"Fine-tuning completed. Model saved at: {finetuned_path}")
    
    logging.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()