import os
import logging
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

from utils.utils import apply_spectral_spatial_mask, segment_and_permute_spectra, calculate_metrics, get_curriculum_segments
from utils.losses import CombinedSSLLoss, SOCRegressionLoss
from viz import (
    save_reconstruction_visualization,
    save_permutation_confusion_matrix,
    save_spectral_signature_comparison
)

      
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available, WandB logging will be disabled")

class BaseTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup output directory
        self.output_dir = Path(config.get('output_dir', './output'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Common training parameters
        self.max_epochs = config.get('max_epochs', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.save_best_model = config.get('save_best_model', True)
        
        # Initialize training state variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = None
        self.no_improvement_epochs = 0
        
    def _save_checkpoint(self, path=None, metrics=None, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics
        }
        
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        if path is None:
            # Determine checkpoint filename based on trainer type
            trainer_type = self.__class__.__name__.lower().replace('trainer', '')
            checkpoint_path = self.checkpoint_dir / f"{trainer_type}_latest.pth"
        else:
            checkpoint_path = path
            
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best and self.save_best_model:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
            # Update config with best model path
            self.config['pretrained_model_path'] = str(best_path)
            logging.info(f"Best model saved to {best_path}")
    
    def save_final_config(self):
        """Save the final config (with pretrained_model_path updated) to the output dir only once."""
        import yaml
        config_file = self.output_dir / "config_used.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logging.info(f"Final configuration saved to {config_file}")
    
    def check_for_checkpoint(self):
        """Check if there's a checkpoint available and load it."""
        # Determine checkpoint filename based on trainer type
        trainer_type = self.__class__.__name__.lower().replace('trainer', '')
        checkpoint_path = self.checkpoint_dir / f"{trainer_type}_latest.pth"
        
        if checkpoint_path.exists():
            logging.info(f"Found checkpoint at {checkpoint_path}")
            self.resume_from_checkpoint(checkpoint_path)
            return True
        else:
            logging.info(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            return False
        
    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metrics = checkpoint.get('best_val_metrics', None)
        
        logging.info(f"Resumed training from checkpoint: {checkpoint_path}")
        logging.info(f"Starting from epoch {self.current_epoch}, global step {self.global_step}")
        logging.info(f"Best validation loss so far: {self.best_val_loss:.6f}")

class PretrainingTrainer(BaseTrainer):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        
        # Pretraining specific parameters
        self.learning_rate = float(config.get('pretraining_learning_rate', 1e-4))
        self.mask_ratio = config.get('mask_ratio', 0.6)
        self.block_size = config.get('block_size', 8)
        self.lambda_perm = config.get('lambda_perm', 0.5)
        self.rec_loss_type = config.get('rec_loss_type', 'mse')
        
        # Read task configuration
        self.use_reconstruction = config.get('use_reconstruction', True)
        self.use_permutation = config.get('use_permutation', True)
        self.rec_loss_scale = config.get('rec_loss_scale', 1e-5)
        
        self.scheduler_enabled = config.get('scheduler_enabled', True)
        
        # Frequency parameters
        self.val_frequency = config.get('validation_frequency', 1)
        self.visualization_frequency = config.get('visualization_frequency', 5)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize scheduler if enabled
        if self.scheduler_enabled:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            self.scheduler = None
        
        # Initialize loss function
        self.criterion = CombinedSSLLoss(
            lambda_perm=self.lambda_perm,
            rec_loss_type=self.rec_loss_type,
            rec_loss_scale=self.rec_loss_scale
        )
        
        # Initialize metrics tracking
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0
        
        # Curriculum learning parameters
        self.use_curriculum = config.get('use_curriculum', False)
        self.min_segments = config.get('min_segments', 3)
        self.max_segments = config.get('max_segments', 8)
        
        # Add embedding extraction to model
        # self.model = add_embedding_extraction_to_model(self.model)
        
        # Metrics history for visualizations
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rec_loss': [],
            'val_rec_loss': [],
            'train_perm_loss': [],
            'val_perm_loss': [],
            'train_perm_acc': [],
            'val_perm_acc': []
        }
        
        # Initialize WandB logging flag
        self.use_wandb_logging = config.get('use_wandb_logging', False)
    
    def train(self, data_loaders):
        """
        Train the model for the specified number of epochs.
        Automatically loads from latest checkpoint if available.
        
        Args:
            data_loaders: Dictionary containing 'train' and 'val' data loaders
        
        Returns:
            Path to the best model checkpoint
        """
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']

        
        # Attempt to load the latest checkpoint if available
        self.check_for_checkpoint()
        
        logging.info(f"Starting pretraining from epoch {self.current_epoch+1} to {self.max_epochs}")
        
        # Track whether curriculum scheduler has been initialized
        if not hasattr(self, 'curriculum_scheduler'):
            self.curriculum_scheduler = None
            # Initialize curriculum scheduler if enabled
            if self.use_curriculum:
                from utils.utils import CurriculumScheduler
                self.curriculum_scheduler = CurriculumScheduler(
                    train_loader,
                    initial_difficulty=self.config.get('curriculum_learning', {}).get('initial_difficulty', 0.2),
                    final_difficulty=self.config.get('curriculum_learning', {}).get('final_difficulty', 1.0),
                    epochs_to_max_difficulty=self.config.get('curriculum_learning', {}).get('epochs_to_max_difficulty', 50)
                )
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Update curriculum difficulty if enabled
            if self.curriculum_scheduler is not None:
                new_difficulty = self.curriculum_scheduler.step(epoch)
                logging.info(f"Epoch {epoch+1}: Curriculum difficulty set to {new_difficulty:.2f}")
            
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            train_loss = train_metrics['loss']
            
            # Extract and store specific metrics for visualization
            self.metrics_history['train_loss'].append(train_loss)
            if 'rec_loss' in train_metrics:
                self.metrics_history['train_rec_loss'].append(train_metrics['rec_loss'])
            if 'perm_loss' in train_metrics:
                self.metrics_history['train_perm_loss'].append(train_metrics['perm_loss'])
            if 'perm_acc' in train_metrics:
                self.metrics_history['train_perm_acc'].append(train_metrics['perm_acc'])
            
            # Log epoch metrics
            logging.info(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}")
            if 'rec_loss' in train_metrics and 'perm_loss' in train_metrics:
                logging.info(f"  Rec Loss: {train_metrics['rec_loss']:.4f}, "
                            f"Perm Loss: {train_metrics['perm_loss']:.4f}")
            
            # Evaluate on validation set
            if (epoch + 1) % self.val_frequency == 0:
                val_metrics = self._validate(val_loader, epoch)
                val_loss = val_metrics['loss']
                
                # Extract and store validation metrics
                self.metrics_history['val_loss'].append(val_loss)
                if 'rec_loss' in val_metrics:
                    self.metrics_history['val_rec_loss'].append(val_metrics['rec_loss'])
                if 'perm_loss' in val_metrics:
                    self.metrics_history['val_perm_loss'].append(val_metrics['perm_loss'])
                if 'perm_acc' in val_metrics:
                    self.metrics_history['val_perm_acc'].append(val_metrics['perm_acc'])
                
                # Log validation metrics
                logging.info(f"Validation Loss: {val_loss:.4f}")
                if 'rec_loss' in val_metrics and 'perm_loss' in val_metrics:
                    logging.info(f"  Val Rec Loss: {val_metrics['rec_loss']:.4f}, "
                                f"Val Perm Loss: {val_metrics['perm_loss']:.4f}")
                
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    improvement = (self.best_val_loss - val_loss) / self.best_val_loss * 100
                    logging.info(f"Validation loss improved from {self.best_val_loss:.6f} to {val_loss:.6f} "
                            f"({improvement:.2f}% improvement)")
                    
                    self.best_val_loss = val_loss
                    self.no_improvement_epochs = 0
                    
                    # Save best model
                    if self.save_best_model:
                        self._save_checkpoint(is_best=True)
                else:
                    self.no_improvement_epochs += 1
                    logging.info(f"No improvement in validation loss for {self.no_improvement_epochs} epochs")
                
                # Step scheduler if it's ReduceLROnPlateau
                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
            
            # Step scheduler if it's not ReduceLROnPlateau
            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Save checkpoint
            self._save_checkpoint()
            
            # Early stopping
            if self.no_improvement_epochs >= self.early_stopping_patience:
                logging.info(f"Early stopping after {self.no_improvement_epochs} epochs without improvement")
                break
            
            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch+1}/{self.max_epochs} completed in {epoch_time:.2f}s")
        
        
        logging.info(f"Training completed after {self.current_epoch+1} epochs")
        logging.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Cleanup WandB
        if hasattr(self, 'use_wandb_logging') and self.use_wandb_logging and 'wandb' in globals():
            wandb.finish()
        
        # Return the path to the best model
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        self.save_final_config()
        return best_model_path
    
    
    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        metrics = {'loss': 0, 'rec_loss': 0, 'perm_loss': 0, 'perm_acc': 0}
        num_batches = 0
        
        # Calculate number of segments if using curriculum
        if self.use_curriculum:
            num_segments = get_curriculum_segments(
                epoch, self.max_epochs, self.min_segments, self.max_segments
            )
        else:
            num_segments = self.config.get('num_segments', 8)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Get input data
            x = batch['data'].to(self.device)
            
            # Normalize the input data if not already normalized
            if x.max() > 10.0:  # Simple heuristic to check if data needs normalization
                x = (x - x.mean()) / (x.std() + 1e-8)
            
            # Determine which tasks to perform
            task = []
            if self.use_reconstruction:
                task.append('reconstruction')
            if self.use_permutation:
                task.append('permutation')
            task = 'both' if len(task) > 1 else (task[0] if task else None)
            
            if task is None:
                logging.warning("No SSL tasks enabled. Enable at least one task.")
                return {'loss': 0}
            
            # Apply masking for reconstruction
            if self.use_reconstruction:
                x_masked, mask = apply_spectral_spatial_mask(
                    x, mask_ratio=self.mask_ratio, block_size=self.block_size
                )
            else:
                x_masked, mask = x, None
            
            # Apply permutation if enabled
            if self.use_permutation:
                x_permuted, perm_targets = segment_and_permute_spectra(x_masked, num_segments)
                perm_targets = {k: v.to(self.device) for k, v in perm_targets.items()}
            else:
                x_permuted = x_masked
                perm_targets = {}
            
            # Forward pass
            outputs = self.model(x_permuted, task=task)
            
            # Calculate loss
            loss, batch_metrics = self.criterion(outputs, perm_targets, mask, x)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate permutation accuracy if applicable
            perm_acc = 0
            if self.use_permutation and 'permutation' in outputs:
                perm_preds = outputs['permutation']
                perm_targets_tensor = perm_targets['inverse_permutation']
                
                # Get predicted indices
                pred_indices = torch.argmax(perm_preds, dim=1)  # [B]
                
                # Calculate accuracy
                correct = (pred_indices == perm_targets_tensor).float()
                perm_acc = torch.mean(correct).item()
                
                # Add to batch metrics
                batch_metrics['perm_acc'] = perm_acc
            
            # Update metrics
            num_batches += 1
            total_loss += loss.item()
            metrics['loss'] = total_loss / num_batches
            
            for k, v in batch_metrics.items():
                if k in metrics:
                    v_val = v.item() if isinstance(v, torch.Tensor) else v
                    metrics[k] = (metrics[k] * (num_batches - 1) + v_val) / num_batches
        
        return metrics
    
    def _validate(self, val_loader, epoch):
        """
        Validate the model on a dataset. Enhanced version with quantitative metrics.
        """
        self.model.eval()
        total_loss = 0
        metrics = {'loss': 0, 'rec_loss': 0, 'perm_loss': 0, 'perm_acc': 0, 
                'rec_mse': 0, 'rec_rmse': 0, 'rec_psnr': 0, 'rec_similarity': 0}
        num_batches = 0
        
        # Determine which tasks to perform
        task = []
        if self.use_reconstruction:
            task.append('reconstruction')
        if self.use_permutation:
            task.append('permutation')
        task = 'both' if len(task) > 1 else (task[0] if task else None)
        
        if task is None:
            logging.warning("No SSL tasks enabled. Enable at least one task.")
            return {'loss': 0}
        
        # Calculate number of segments if using curriculum
        if self.use_curriculum:
            num_segments = get_curriculum_segments(
                epoch, self.max_epochs, self.min_segments, self.max_segments
            )
        else:
            num_segments = self.config.get('num_segments', 8)
        
        # Store samples for visualization
        vis_samples = {
            'original': None,
            'masked': None,
            'reconstructed': None,
            'shuffled': None,
            'perm_preds': None,
            'perm_targets': None
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                # Get input data
                x = batch['data'].to(self.device)
                
                # Normalize the input data if not already normalized
                if x.max() > 10.0:  # Simple heuristic to check if data needs normalization
                    x = (x - x.mean()) / (x.std() + 1e-8)
                
                # Apply masking for reconstruction
                if self.use_reconstruction:
                    x_masked, mask = apply_spectral_spatial_mask(
                        x, mask_ratio=self.mask_ratio, block_size=self.block_size
                    )
                else:
                    x_masked, mask = x, None
                
                # Apply permutation if enabled
                if self.use_permutation:
                    x_permuted, perm_targets = segment_and_permute_spectra(x_masked, num_segments)
                    perm_targets = {k: v.to(self.device) for k, v in perm_targets.items()}
                else:
                    x_permuted = x_masked
                    perm_targets = {}
                
                # Forward pass
                outputs = self.model(x_permuted, task=task)
                
                # Calculate loss
                loss, batch_metrics = self.criterion(outputs, perm_targets, mask, x)
                
                # Calculate permutation accuracy if applicable
                perm_acc = 0
                if self.use_permutation and 'permutation' in outputs:
                    perm_preds = outputs['permutation']
                    perm_targets_tensor = perm_targets['inverse_permutation']
                    
                    # Get predicted indices
                    pred_indices = torch.argmax(perm_preds, dim=1)  # [B]
                    
                    # Calculate accuracy
                    correct = (pred_indices == perm_targets_tensor).float()
                    perm_acc = torch.mean(correct).item()
                    
                    # Add to batch metrics
                    batch_metrics['perm_acc'] = perm_acc
                    
                    # Calculate additional permutation metrics using our custom functions
                    try:
                        from utils.metrics import calculate_permutation_metrics
                        perm_metrics = calculate_permutation_metrics(perm_preds, perm_targets_tensor)
                        
                        # Add these metrics to batch_metrics
                        for k, v in perm_metrics.items():
                            if k != 'confusion_matrix':  # Skip large objects
                                batch_metrics[f'perm_{k}'] = v
                    except ImportError:
                        logging.warning("metrics module not found, skipping advanced permutation metrics")
                
                # Calculate reconstruction metrics if applicable
                if self.use_reconstruction and 'reconstruction' in outputs:
                    try:
                        from utils.metrics import calculate_reconstruction_metrics
                        rec_metrics = calculate_reconstruction_metrics(x, outputs['reconstruction'], mask)
                        
                        # Add these metrics to batch_metrics
                        for k, v in rec_metrics.items():
                            batch_metrics[f'rec_{k}'] = v
                    except ImportError:
                        logging.warning("metrics module not found, skipping advanced reconstruction metrics")
                
                # Update metrics
                num_batches += 1
                total_loss += loss.item()
                metrics['loss'] = total_loss / num_batches
                
                for k, v in batch_metrics.items():
                    if k in metrics:
                        v_val = v.item() if isinstance(v, torch.Tensor) else v
                        metrics[k] = (metrics[k] * (num_batches - 1) + v_val) / num_batches
                
                # Store first batch for visualization
                if batch_idx == 0:
                    vis_samples['original'] = x.detach()
                    if self.use_reconstruction:
                        vis_samples['masked'] = x_masked.detach()
                        if 'reconstruction' in outputs:
                            vis_samples['reconstructed'] = outputs['reconstruction'].detach()
                    if self.use_permutation:
                        vis_samples['shuffled'] = x_permuted.detach()
                        vis_samples['perm_targets'] = perm_targets['inverse_permutation'].detach()
                        if 'permutation' in outputs:
                            vis_samples['perm_preds'] = outputs['permutation'].detach()
            
            # Log detailed metrics
            for k, v in metrics.items():
                if k not in ['loss', 'rec_loss', 'perm_loss'] and isinstance(v, (int, float)):
                    logging.info(f"Validation metric {k}: {v:.4f}")
        
        # Store visualization samples for later use
        self.vis_samples = vis_samples
        
        if (epoch + 1) % self.visualization_frequency == 0:
            # Reconstruction visualization
            if self.use_reconstruction and 'reconstructed' in self.vis_samples:
                save_reconstruction_visualization(
                    self.vis_samples['original'],
                    self.vis_samples['masked'],
                    self.vis_samples['reconstructed'],
                    self.output_dir,
                    epoch=epoch+1
                )

            # Permutation confusion matrix visualization
            if self.use_permutation and 'perm_preds' in self.vis_samples:
                from utils.metrics import calculate_permutation_metrics
                perm_metrics = calculate_permutation_metrics(
                    self.vis_samples['perm_preds'],
                    self.vis_samples['perm_targets']
                )
                save_permutation_confusion_matrix(
                    perm_metrics['confusion_matrix'],
                    self.output_dir,
                    epoch=epoch+1
                )

            # Spectral signature visualization
            if all(k in self.vis_samples for k in ['original', 'shuffled', 'reconstructed', 'true_perm', 'pred_perm']):
                save_spectral_signature_comparison(
                    self.vis_samples['original'],
                    self.vis_samples['shuffled'],
                    self.vis_samples['reconstructed'],
                    self.vis_samples['true_perm'],
                    self.vis_samples['pred_perm'],
                    self.output_dir,
                    epoch=epoch+1
                )
        
        return metrics
    
class FinetuningTrainer(BaseTrainer):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        
        # Finetuning specific parameters
        self.learning_rate = float(config.get('finetuning_learning_rate', 5e-5))
        self.weight_decay = float(config.get('weight_decay', 1e-5))
        self.loss_type = config.get('loss_type', 'mse')
        self.scheduler_enabled = config.get('scheduler_enabled', True)
        
        # Initialize optimizer with different learning rates for frozen vs trainable parts
        if hasattr(model, 'encoder') and hasattr(model, 'regression_head'):
            # The model has separate encoder and head components
            encoder_params = []
            head_params = []
            
            # Group parameters by component
            for name, param in model.named_parameters():
                if name.startswith('encoder'):
                    encoder_params.append(param)
                else:
                    head_params.append(param)
            
            # Create optimizer with parameter groups
            self.optimizer = optim.Adam([
                {'params': encoder_params, 'lr': self.learning_rate * 0.1},
                {'params': head_params, 'lr': self.learning_rate}
            ], weight_decay=self.weight_decay)
        else:
            # Standard optimizer for the whole model
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Initialize scheduler if enabled
        if self.scheduler_enabled:
            scheduler_type = config.get('scheduler_type', 'plateau')
            if scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.max_epochs
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # Initialize loss function
        self.criterion = SOCRegressionLoss(loss_type=self.loss_type)
        
        # Initialize metrics tracking
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0
        self.use_wandb_logging = config.get('use_wandb_logging', False)
        
        # History for visualization
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'train_rpd': [],
            'val_rpd': []
        }
        
    def train(self, train_loader, val_loader):
        logging.info(f"Starting fine-tuning for {self.max_epochs} epochs")
        
        # Create visualization directory
        # vis_dir = create_visualization_dir(self.output_dir)
        
        # Resume from checkpoint if available
        resume_finetuning = self.config.get('resume_finetuning', False)
        if resume_finetuning:
            self.check_for_checkpoint()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss, val_metrics = self._validate(val_loader)
            
            # Store metrics for visualization
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            
            if train_metrics:
                self.metrics_history['train_rmse'].append(train_metrics.get('rmse', 0))
                self.metrics_history['train_r2'].append(train_metrics.get('r2', 0))
                self.metrics_history['train_rpd'].append(train_metrics.get('rpd', 0))
            
            if val_metrics:
                self.metrics_history['val_rmse'].append(val_metrics.get('rmse', 0))
                self.metrics_history['val_r2'].append(val_metrics.get('r2', 0))
                self.metrics_history['val_rpd'].append(val_metrics.get('rpd', 0))
            
            # Log metrics
            logging.info(f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}")
            
            if val_metrics is not None:
                logging.info(f"  Val RMSE: {val_metrics['rmse']:.4f}, "
                            f"Val R²: {val_metrics['r2']:.4f}, "
                            f"Val RPD: {val_metrics['rpd']:.4f}")
            
            # Generate metrics visualization periodically
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._visualize_metrics(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"finetune_latest.pth"
            self._save_checkpoint(checkpoint_path, val_metrics)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                improvement = (self.best_val_loss - val_loss) / self.best_val_loss * 100
                logging.info(f"Validation loss improved from {self.best_val_loss:.6f} to {val_loss:.6f} "
                           f"({improvement:.2f}% improvement)")
                
                self.best_val_loss = val_loss
                self.no_improvement_epochs = 0
                
                # Save best model
                self._save_checkpoint(checkpoint_path, val_metrics, is_best=True)
            else:
                self.no_improvement_epochs += 1
                logging.info(f"No improvement in validation loss for {self.no_improvement_epochs} epochs")
            
            # Early stopping
            if self.no_improvement_epochs >= self.early_stopping_patience:
                logging.info(f"Early stopping after {epoch+1} epochs")
                break
            
            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch completed in {epoch_time:.2f}s")
        
        # Final visualization
        self._visualize_metrics(epoch, is_final=True)
        
        # Save final config
        self.save_final_config()
        
        return self.checkpoint_dir / 'best_model.pth'
    
    def _train_epoch(self, train_loader):
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Process outputs
            # If outputs is [B, 1, H, W], take mean across spatial dimensions
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=(-1, -2))
            
            # Squeeze if needed to match targets
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.cpu())
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        # Stack predictions and targets for metrics calculation
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _validate(self, val_loader):
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Process outputs
                # If outputs is [B, 1, H, W], take mean across spatial dimensions
                if outputs.dim() > 2:
                    outputs = outputs.mean(dim=(-1, -2))
                
                # Squeeze if needed to match targets
                outputs = outputs.squeeze()
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        # Stack predictions and targets for metrics calculation
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _visualize_metrics(self, epoch, is_final=False):
        """
        Visualize training and validation metrics.
        
        Args:
            epoch: Current epoch
            is_final: Whether this is the final visualization
        """
        vis_dir = self.output_dir / 'visualizations' / 'fine_tuning'
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot RMSE
        axes[0, 1].plot(self.metrics_history['train_rmse'], label='Train')
        axes[0, 1].plot(self.metrics_history['val_rmse'], label='Validation')
        axes[0, 1].set_title('RMSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot R²
        axes[1, 0].plot(self.metrics_history['train_r2'], label='Train')
        axes[1, 0].plot(self.metrics_history['val_r2'], label='Validation')
        axes[1, 0].set_title('R²')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot RPD
        axes[1, 1].plot(self.metrics_history['train_rpd'], label='Train')
        axes[1, 1].plot(self.metrics_history['val_rpd'], label='Validation')
        axes[1, 1].set_title('RPD (Ratio of Performance to Deviation)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RPD')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Set a title for the entire figure
        fig.suptitle(f'Fine-tuning Metrics (Epoch {epoch+1})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the figure
        filename = 'final_metrics.png' if is_final else f'metrics_epoch{epoch+1}.png'
        plt.savefig(vis_dir / filename, dpi=150)
        plt.close(fig)
        
        logging.info(f"Saved fine-tuning metrics visualization to {vis_dir / filename}")