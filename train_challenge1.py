#!/usr/bin/env python3
"""
Training Script for Challenge 1: Cross-Task Transfer Learning
EEG Foundation Challenge 2025

Supports both CNN and Transformer encoders with architecture-specific optimizations.

Usage:
    python train_challenge1.py --config src/configs/challenge1_config.yaml
    python train_challenge1.py --encoder cnn
    python train_challenge1.py --encoder transformer
    
Author: EEG Foundation Challenge Team
Date: 2025
License: MIT
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.challenge1_baseline import Challenge1Model
from data.dataset_loader import create_challenge1_dataloaders
from utils.gpu_utils import get_recommended_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('challenge1_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup for transformer training.
    
    Gradually increases learning rate from warmup_start_lr to base_lr over warmup_epochs,
    then applies cosine annealing schedule. This is particularly important for transformer
    architectures which benefit from gradual learning rate warmup.
    
    Args:
        optimizer: PyTorch optimizer instance
        warmup_epochs: Number of epochs for warmup phase
        warmup_start_lr: Starting learning rate for warmup
        base_lr: Target learning rate after warmup
        total_epochs: Total number of training epochs
        last_epoch: Index of last epoch (for resuming training)
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float,
        base_lr: float,
        total_epochs: int,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Compute learning rate based on current epoch.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * warmup_factor 
                    for _ in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.base_lr * cosine_factor for _ in self.base_lrs]

class Challenge1Trainer:
    """
    Trainer class for Challenge 1 with support for both CNN and Transformer encoders.
    
    This class handles the complete training pipeline for the per-trial approach,
    including data loading, model creation, training, validation, and evaluation.
    
    Attributes:
        config: Configuration dictionary
        encoder_type: Type of encoder ('cnn' or 'transformer')
        gpu_config: GPU configuration information
        gpus: List of available GPUs
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
        """
        Initialize Challenge 1 trainer.
        
        Args:
            config: Configuration dictionary from YAML file containing all
                   training parameters, model configuration, and data settings.
            checkpoint_path: Optional path to checkpoint file for resuming training
                   
        Raises:
            ValueError: If configuration is invalid or missing required keys
            RuntimeError: If GPU requirements are not met
        """
        self.config = config
        self.encoder_type = config['model'].get('encoder_type', 'cnn')
        self.checkpoint_path = checkpoint_path
        
        # GPU setup first
        self.gpu_config = get_recommended_config()  # Call without model initially
        self.gpus = self.gpu_config['gpu_info']
        
        # Validate configuration
        self._validate_config()
        
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()
        
        # Check GPU memory requirements
        self._check_gpu_requirements()

    
    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary.
        
        Raises:
            ValueError: If configuration is missing required keys or has invalid values
        """
        required_keys = ['model', 'data', 'training', 'hardware', 'logging']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate model configuration
        if 'encoder_type' not in self.config['model']:
            raise ValueError("Missing 'encoder_type' in model configuration")
        
        if self.config['model']['encoder_type'] not in ['cnn', 'transformer']:
            raise ValueError("encoder_type must be 'cnn' or 'transformer'")
        
        # Validate data configuration
        if 'data_dir' not in self.config['data']:
            raise ValueError("Missing 'data_dir' in data configuration")
        
        # Validate training configuration
        if 'max_epochs' not in self.config['training']:
            raise ValueError("Missing 'max_epochs' in training configuration")
        
        logger.info("✓ Configuration validation passed")
        logger.info(f"Available GPUs: {self.gpus['num_gpus']}")
        if self.gpus['cuda_available']:
            logger.info(f"GPU Memory: {[f'{mem:.1f}GB' for mem in self.gpus['gpu_memory']]}")
        logger.info("="*70)
        
        # Print architecture-specific information
        if self.encoder_type == 'transformer':
            logger.info("TRANSFORMER ARCHITECTURE:")
            transformer_config = self.config['model']['transformer']
            logger.info(f"   • Layers: {transformer_config['num_layers']}")
            logger.info(f"   • Attention Heads: {transformer_config['nhead']}")
            logger.info(f"   • Embedding Dim: {transformer_config['d_model']}")
            logger.info(f"   • Patch Size: {transformer_config['patch_size']}")
            logger.info(f"   • Dropout: {transformer_config['dropout']}")
        else:
            logger.info("CNN ARCHITECTURE:")
            cnn_config = self.config['model']['cnn']
            logger.info(f"   • Temporal Filters: {cnn_config['temporal_filters']}")
            logger.info(f"   • Spatial Filters: {cnn_config['spatial_filters']}")
            logger.info(f"   • Dropout: {cnn_config['dropout']}")
        logger.info("")
        
    def _check_gpu_requirements(self):
        """Check if available GPU memory meets architecture requirements"""
        if not self.gpus['cuda_available']:
            logger.warning("No GPUs detected. Training will be slow on CPU.")
            return
        
        # Get memory requirements from config
        required_memory = self.config['hardware'].get(f'{self.encoder_type}_memory_gb', 8)
        
        for i, (gpu_name, gpu_memory) in enumerate(zip(self.gpus['gpu_names'], self.gpus['gpu_memory'])):
            available_gb = int(gpu_memory)
            if available_gb < required_memory:
                logger.warning(f"GPU {i} ({gpu_name}) has {available_gb}GB but {self.encoder_type.upper()} needs {required_memory}GB")
                logger.warning("Consider reducing batch size or using CNN encoder")
            else:
                logger.info(f"GPU {i} ({gpu_name}): {available_gb}GB (need {required_memory}GB)")
        
    def setup_directories(self):
        """Setup necessary directories"""
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup experiment logging"""
        experiment_name = f"{self.config['logging']['experiment_name']}_{self.encoder_type}"
        project_name = self.config['logging']['project_name']
        
        # WandB logger (if available)
        try:
            self.wandb_logger = WandbLogger(
                name=experiment_name,
                project=project_name,
                save_dir='logs',
                config=self.config
            )
            logger.info("WandB logging initialized")
        except Exception as e:
            logger.warning(f"WandB not available: {e}")
            self.wandb_logger = None
        
        # TensorBoard logger (fallback)
        self.tb_logger = TensorBoardLogger(
            save_dir='logs',
            name=experiment_name,
            version=None
        )
        
    def create_model(self) -> Challenge1Model:
        """Create Challenge 1 model with proper configuration"""
        # Base model configuration
        model_config = {
            'n_channels': self.config['model']['n_channels'],
            'n_times': self.config['model']['n_times'],
            'hidden_dim': self.config['model']['hidden_dim'],
            'encoder_type': self.encoder_type,
            'learning_rate': float(self.config['training']['learning_rate']),
            'weight_decay': float(self.config['training']['weight_decay']),
            'min_lr': float(self.config['training']['min_lr']),
            'max_epochs': self.config['training']['max_epochs'],
            'mixed_precision': self.config['training']['mixed_precision'],
            **self.config['training']['loss_weights']
        }
        
        # Add encoder-specific configurations
        if self.encoder_type == 'transformer':
            model_config.update(self.config['model']['transformer'])
        else:
            model_config.update(self.config['model']['cnn'])
        
        # Add scheduler configuration
        model_config['scheduler_config'] = self.config['training']['scheduler']
        
        model = Challenge1Model(model_config)
        
        # Update GPU configuration with actual model
        if self.gpus['cuda_available']:
            self.gpu_config = get_recommended_config(
                model=model,
                sequence_length=self.config['model']['n_times'],
                num_channels=self.config['model']['n_channels']
            )
        
        # Print model architecture
        logger.info(f"\n  MODEL ARCHITECTURE:")
        logger.info(f"   • Encoder Type: {self.encoder_type.upper()}")
        logger.info(f"   • Shared Encoder: {model.shared_encoder.hidden_dim}D features")
        logger.info(f"   • Age Head: Regression (1 output)")
        logger.info(f"   • Sex Head: Classification (2 classes)")
        logger.info(f"   • Response Time Head: Regression (1 output)")
        logger.info(f"   • Hit/Miss Head: Classification (2 classes)")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"   • Total Parameters: {total_params:,}")
        logger.info(f"   • Trainable Parameters: {trainable_params:,}")
        
        # Memory estimation
        model_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32 parameter
        logger.info(f"   • Model Size: {model_size_mb:.1f} MB")
        
        # Show GPU-optimized batch size if available
        if self.gpus['cuda_available']:
            logger.info(f"   • GPU-Optimized Batch Size: {self.gpu_config['batch_size']}")
        
        return model
    
    def create_callbacks(self) -> List[pl.Callback]:
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['logging']['checkpoint_dir'],
            filename=f'challenge1_{self.encoder_type}' + '-{epoch:02d}-{val/total_loss:.3f}',
            monitor=self.config['logging']['monitor'],
            mode='min',
            save_top_k=self.config['logging']['save_top_k'],
            save_last=self.config['logging']['save_last'],
            save_weights_only=self.config['logging']['save_weights_only'],
            verbose=True,
            save_on_train_epoch_end=self.config['logging']['save_on_train_epoch_end'],  # Only save at validation end, not every epoch
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config['training']['early_stopping']['monitor'],
            patience=self.config['training']['early_stopping']['patience'],
            mode=self.config['training']['early_stopping']['mode'],
            min_delta=self.config['training']['early_stopping']['min_delta'],
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def create_trainer(self, callbacks: List[pl.Callback]) -> pl.Trainer:
        """Create PyTorch Lightning trainer"""
        # Architecture-specific batch size
        if self.encoder_type == 'transformer' and 'transformer_config' in self.config.get('comparison', {}):
            batch_size = self.config['comparison']['transformer_config']['batch_size']
        elif self.encoder_type == 'cnn' and 'cnn_config' in self.config.get('comparison', {}):
            batch_size = self.config['comparison']['cnn_config']['batch_size']
        else:
            batch_size = self.config['training']['batch_size']
        
        # Handle deterministic setting for transformer (adaptive pooling doesn't have deterministic CUDA implementation)
        deterministic_setting = self.config['deterministic']
        if self.config['model']['encoder_type'] == 'transformer':
            deterministic_setting = False  # Disable strict determinism for transformer
            logger.info("Disabling strict determinism for transformer encoder due to adaptive pooling operations")
        
        trainer_config = {
            'max_epochs': self.config['training']['max_epochs'],
            'gradient_clip_val': self.config['training']['gradient_clip_val'],
            'accumulate_grad_batches': self.config['training']['accumulate_grad_batches'],
            'deterministic': deterministic_setting,
            'log_every_n_steps': self.config['logging']['log_every_n_steps'],
            'val_check_interval': self.config['validation']['val_check_interval'],
            'callbacks': callbacks,
            'logger': [self.tb_logger] + ([self.wandb_logger] if self.wandb_logger else []),
            'enable_checkpointing': True,
            'enable_progress_bar': True,
            'enable_model_summary': True
        }
        
        # GPU configuration
        logger.info(f"GPU Info: {self.gpus}")
        logger.info(f"Debug mode: {self.config.get('debug', False)}")
        
        if self.gpus['cuda_available']:
            # In debug mode, use single GPU to avoid distributed training issues
            if self.config.get('debug', False):
                trainer_config.update({
                    'accelerator': 'gpu',
                    'devices': 1,  # Single GPU for debug
                    'precision': self.config['hardware']['precision']
                })
                logger.info("Debug mode: Using single GPU")
            else:
                trainer_config.update({
                    'accelerator': 'gpu',
                    'devices': self.gpus['num_gpus'],
                    'precision': self.config['hardware']['precision']
                })
                
                # Multi-GPU strategy
                if self.gpus['num_gpus'] > 1:
                    trainer_config['strategy'] = 'ddp_find_unused_parameters_true'
                    logger.info(f"Using DDP strategy with {self.gpus['num_gpus']} GPUs (find_unused_parameters=True)")
                else:
                    logger.info(f"Single GPU detected: {self.gpus['num_gpus']} GPU")
        else:
            trainer_config.update({
                'accelerator': 'cpu',
                'devices': 1
            })
            logger.info("Using CPU training")
        
        logger.info(f"Final trainer config: {trainer_config}")
        
        trainer = pl.Trainer(**trainer_config)
        
        return trainer
    
    def evaluate_model(self, model: Challenge1Model, val_loader) -> Dict[str, float]:
        """
        Evaluate the trained model on validation set
        
        Args:
            model: Trained Challenge 1 model
            val_loader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Model should already be in eval mode and on correct device
        model.eval()
        
        all_predictions = {
            'response_time': [],
            'hit_miss': []
        }
        
        all_targets = {
            'response_time': [],
            'hit_miss': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                input_features, targets = batch
                
                # Move to device
                if torch.cuda.is_available():
                    # Handle both dictionary and tensor formats
                    if isinstance(input_features, dict):
                        input_features = {k: v.cuda() for k, v in input_features.items()}
                    else:
                        input_features = input_features.cuda()
                    targets = {k: v.cuda() for k, v in targets.items()}
                
                # Forward pass
                predictions = model(input_features)
                
                # Collect predictions and targets
                for key in all_predictions.keys():
                    if key in predictions:
                        if key == 'hit_miss':
                            # Classification: get predicted class
                            pred_class = torch.argmax(predictions[key], dim=1)
                            all_predictions[key].extend(pred_class.cpu().numpy())
                            all_targets[key].extend(targets[key].cpu().numpy())
                        else:
                            # Regression: get predicted value
                            all_predictions[key].extend(predictions[key].squeeze().cpu().numpy())
                            all_targets[key].extend(targets[key].squeeze().cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        # Response time metrics (regression) - PRIMARY METRIC
        if all_targets['response_time']:
            # Apply numerical stability checks
            rt_targets = np.array(all_targets['response_time'])
            rt_predictions = np.array(all_predictions['response_time'])
            
            # Check for invalid values
            if np.any(np.isnan(rt_predictions)) or np.any(np.isinf(rt_predictions)):
                logger.warning(f"Invalid predictions detected: NaN={np.any(np.isnan(rt_predictions))}, Inf={np.any(np.isinf(rt_predictions))}")
                rt_predictions = np.where(np.isfinite(rt_predictions), rt_predictions, np.median(rt_targets))
            
            if np.any(np.isnan(rt_targets)) or np.any(np.isinf(rt_targets)):
                logger.warning(f"Invalid targets detected: NaN={np.any(np.isnan(rt_targets))}, Inf={np.any(np.isinf(rt_targets))}")
                rt_targets = np.where(np.isfinite(rt_targets), rt_targets, np.median(rt_targets))
            
            # Ensure both arrays have finite values before computing metrics
            if np.all(np.isfinite(rt_targets)) and np.all(np.isfinite(rt_predictions)):
                rt_r2 = r2_score(rt_targets, rt_predictions)
                rt_mae = mean_absolute_error(rt_targets, rt_predictions)
                rt_rmse = np.sqrt(mean_squared_error(rt_targets, rt_predictions))
                
                metrics['response_time_r2'] = rt_r2
                metrics['response_time_mae'] = rt_mae
                metrics['response_time_rmse'] = rt_rmse
            else:
                logger.warning("Cannot compute metrics due to invalid values")
                metrics['response_time_r2'] = 0.0
                metrics['response_time_mae'] = float('inf')
                metrics['response_time_rmse'] = float('inf')
        
        # Hit/miss metrics (classification)
        if all_targets['hit_miss']:
            hm_acc = accuracy_score(all_targets['hit_miss'], all_predictions['hit_miss'])
            metrics['hit_miss_accuracy'] = hm_acc
        
        return metrics
    
    def train(self):
        # Create callbacks and trainer
        callbacks = self.create_callbacks()
        trainer = self.create_trainer(callbacks)
        
        if trainer.global_rank == 0:
            logger.info("="*70)
            logger.info("EEG Foundation Challenge 1: Cross-Task Transfer Learning")
            logger.info("Official Task: Predicting CCD behavioral outcomes from primary CCD EEG input.")
            logger.info("="*70)
            logger.info(f"Encoder Architecture: {self.encoder_type.upper()}")
        logger.info("Starting Challenge 1 training...")
        
        # Architecture-specific batch size
        if self.encoder_type == 'transformer' and 'transformer_config' in self.config.get('comparison', {}):
            batch_size = self.config['comparison']['transformer_config']['batch_size']
        elif self.encoder_type == 'cnn' and 'cnn_config' in self.config.get('comparison', {}):
            batch_size = self.config['comparison']['cnn_config']['batch_size']
        else:
            batch_size = self.config['training']['batch_size']
        
        # Create data loaders
        logger.info("Creating Challenge 1 data loaders...")
        
        # Get CCD-only settings from config for clarity
        use_demographics = self.config.get('challenge1', {}).get('use_demographics', False)
        use_sus_eeg = self.config.get('challenge1', {}).get('use_sus_eeg', False)
        
        logger.info(f"Feature Flags: use_demographics={use_demographics}, use_sus_eeg={use_sus_eeg}")
        
        train_loader, val_loader = create_challenge1_dataloaders(
            data_dir=self.config['data']['data_dir'],
            config=self.config,
            batch_size=batch_size,
            num_workers=self.config['hardware']['num_workers'],
            use_demographics=use_demographics,
            use_sus_eeg=use_sus_eeg
        )
        
        # Create model
        logger.info("Creating Challenge 1 model...")
        model = self.create_model()
        
        # Train model
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Resuming training from checkpoint: {self.checkpoint_path}")
            trainer.fit(model, train_loader, val_loader, ckpt_path=self.checkpoint_path)
        else:
            logger.info("Starting new training...")
            trainer.fit(model, train_loader, val_loader)
        
        # Evaluate on validation set (since test set is held-out)
        logger.info("Evaluating on validation set...")
        # Ensure model is in eval mode and on the correct device
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        test_metrics = self.evaluate_model(model, val_loader)
        
        # Log final results
        logger.info("\nTRAINING COMPLETED!")
        logger.info("="*70)
        logger.info(f"FINAL VALIDATION RESULTS ({self.encoder_type.upper()} ENCODER):")
        logger.info("="*70)
        
        # Primary metric (Challenge 1 focus)
        if 'response_time_r2' in test_metrics:
            logger.info(f"Response Time R²: {test_metrics['response_time_r2']:.4f}")
            logger.info(f"   Response Time MAE: {test_metrics['response_time_mae']:.2f} ms")
            logger.info(f"   Response Time RMSE: {test_metrics['response_time_rmse']:.2f} ms")
        
        # Secondary metrics
        if 'hit_miss_accuracy' in test_metrics:
            logger.info(f"Hit/Miss Accuracy: {test_metrics['hit_miss_accuracy']:.4f}")
        
        logger.info("="*70)
        
        # Save final results
        results_file = Path(f'challenge1_{self.encoder_type}_results.yaml')
        test_metrics['encoder_type'] = self.encoder_type
        test_metrics['total_parameters'] = sum(p.numel() for p in model.parameters())
        
        with open(results_file, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        
        logger.info(f"Results saved to: {results_file}")
        
        return model, test_metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Challenge 1 Model')
    parser.add_argument(
        '--config',
        type=str, 
        default='src/configs/challenge1_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        choices=['cnn', 'transformer'],
        help='Encoder type (overrides config)'
    )
    # parser.add_argument(
    #     '--data-dir',
    #     type=str,
    #     default='./data/raw/HBN_BIDS_EEG',
    #     help='Path to HBN_BIDS_EEG dataset'
    # )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='src/data/raw/HBN_BIDS_EEG',
        help='Path to HBN_BIDS_EEG dataset'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced data'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Train both CNN and Transformer for comparison'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override encoder type if provided
    if args.encoder:
        config['model']['encoder_type'] = args.encoder
    
    # Override data directory if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    # Debug mode adjustments
    if args.debug:
        logger.info("Debug mode enabled")
        config['training']['max_epochs'] = 2
        config['training']['batch_size'] = 10
        config['hardware']['num_workers'] = 64
        # Note: Split strategy is hardcoded according to challenge requirements
        # Debug mode will use smaller subset of releases for faster testing
    
    # Set random seeds for reproducibility
    pl.seed_everything(config['seed'])
    
    results = {}
    
    if args.compare or config.get('comparison', {}).get('run_both_architectures', False):
        # Train both architectures for comparison
        logger.info("Running architecture comparison...")
        
        for encoder_type in ['cnn', 'transformer']:
            logger.info(f"\n{'='*20} {encoder_type.upper()} TRAINING {'='*20}")
            
            # Update config for current encoder
            config['model']['encoder_type'] = encoder_type
            
            # Create trainer and train
            trainer = Challenge1Trainer(config, checkpoint_path=args.resume)
            model, metrics = trainer.train()
            results[encoder_type] = metrics
        
        # Compare results
        logger.info("\nARCHITECTURE COMPARISON:")
        logger.info("="*70)
        for encoder, metrics in results.items():
            logger.info(f"{encoder.upper()} ENCODER:")
            logger.info(f"  Response Time R²: {metrics.get('response_time_r2', 0):.4f}")
            logger.info(f"  Hit/Miss Accuracy: {metrics.get('hit_miss_accuracy', 0):.4f}")
            logger.info(f"  Parameters: {metrics.get('total_parameters', 0):,}")
        
        # Save comparison results
        with open('architecture_comparison.yaml', 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
    else:
        # Train single architecture
        trainer = Challenge1Trainer(config, checkpoint_path=args.resume)
        model, results = trainer.train()
    
    logger.info("Challenge 1 training completed successfully!")
    
    return results

if __name__ == '__main__':
    main()