#!/usr/bin/env python3
"""
Challenge 1: Cross-Task Transfer Learning - Shared Encoder + 2 Heads
EEG Foundation Challenge 2025

OFFICIAL CHALLENGE CONSTRAINTS:
==============================
- Training Input: CCD EEG data (X1) as primary input
- Optional Additional Features: SuS EEG data (X2) and demographics/psychopathology (P)
- Prediction Targets: CCD behavioral outcomes per trial:
  * Response time (regression)
  * Hit/miss accuracy (binary classification)
- Constraint: Participants MUST use X1 (CCD EEG) but can choose to use X2 (SuS EEG) and P (demographics/psychopathology) as additional features

Official Challenge 1 Data Structure:
- X1 ∈ ℝ^(c×n×t1): CCD EEG recording (c=128 channels, n≈70 epochs, t1=2 seconds)
- X2 ∈ ℝ^(c×t2): SuS EEG recording (c=128 channels, t2=total time samples)  
- P ∈ ℝ^7: Subject traits (3 demographics + 4 psychopathology factors)

Model Architecture:
==================
                    ┌─ Response Time Head (regression)
CCD EEG (X1) → Encoder → └─ Hit/Miss Head (binary classification)

Optional Additional Features:
- SuS EEG data can be concatenated or used as additional input
- Demographics and psychopathology factors can be used as additional features

Encoder Options:
- CNN-based: Temporal + Spatial convolutions (original)
- Transformer-based: Multi-head self-attention with positional encoding

The shared encoder learns representations from CCD EEG data that can predict
behavioral outcomes on the same task.

Training Strategy:
=================
- Multi-task learning with weighted loss functions
- Shared encoder learns task-invariant representations
- Task-specific heads capture different aspects of cognition
- Optional use of additional features (SuS EEG, demographics) for enhanced performance

Author: EEG Foundation Challenge Team
Date: 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import yaml
import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

import mne
from sklearn.metrics import f1_score, mean_squared_error, r2_score, mean_absolute_error
from utils.gpu_utils import detect_gpus, get_recommended_config


class PositionalEncoding(nn.Module):
    """
    Positional encoding for EEG data with both temporal and spatial dimensions.
    
    For EEG data (n_channels, n_times), we need to encode:
    1. Temporal position: where in time each sample occurs
    2. Spatial position: which channel each sample comes from
    """
    
    def __init__(self, d_model: int, max_len: int = 512, n_channels: int = 128):
        super().__init__()
        self.d_model = d_model
        
        # Temporal positional encoding (standard sinusoidal): (max_len, d_model)
        # Each row corresponds to a time index (0 to max_len-1)
        pe_temporal = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)
        pe_temporal[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model/2)
        pe_temporal[:, 1::2] = torch.cos(position * div_term) # (max_len, d_model/2)
        self.register_buffer('pe_temporal', pe_temporal)
        
        # Spatial positional encoding (for channels): (n_channels, d_model)
        # Each row corresponds to a channel index (0 to n_channels-1)
        pe_spatial = torch.zeros(n_channels, d_model)
        position_spatial = torch.arange(0, n_channels, dtype=torch.float).unsqueeze(1) # (n_channels, 1)
        div_term_spatial = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)
        pe_spatial[:, 0::2] = torch.sin(position_spatial * div_term_spatial) # (n_channels, d_model/2)
        pe_spatial[:, 1::2] = torch.cos(position_spatial * div_term_spatial) # (n_channels, d_model/2)
        self.register_buffer('pe_spatial', pe_spatial)
    
    def forward(self, x_patches: torch.Tensor, x_embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D positional encoding: spatial PE to channels, temporal PE to patches.
        
        Args:
            x_patches: Tensor of shape (batch_size, n_patches, n_channels, patch_size)
            x_embedded: Tensor of shape (batch_size, n_patches, d_model) - after patch embedding
            
        Returns:
            Tensor with added spatial + temporal positional encodings
        """
        batch_size, n_patches, n_channels, patch_size = x_patches.shape
        _, _, d_model = x_embedded.shape
        
        # Bounds checking
        if n_patches > self.pe_temporal.shape[0]:
            raise ValueError(f"n_patches ({n_patches}) exceeds max_len ({self.pe_temporal.shape[0]})")
        if n_channels > self.pe_spatial.shape[0]:
            raise ValueError(f"n_channels ({n_channels}) exceeds spatial max ({self.pe_spatial.shape[0]})")
        
        # 1. Apply spatial PE to channels within each patch
        # Get spatial PE for each channel: (n_channels, d_model)
        pe_spatial = self.pe_spatial[:n_channels]  # (n_channels, d_model)
        
        # Expand to match patch structure: (batch_size, n_patches, n_channels, d_model)
        pe_spatial_expanded = pe_spatial.unsqueeze(0).unsqueeze(0).expand(batch_size, n_patches, -1, -1)
        
        # 2. Apply temporal PE to patches
        # Get temporal PE for each patch: (n_patches, d_model)  
        pe_temporal = self.pe_temporal[:n_patches]  # (n_patches, d_model)
        
        # Expand to match structure: (batch_size, n_patches, d_model)
        pe_temporal_expanded = pe_temporal.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Combine spatial and temporal PE
        # Average the two encodings to avoid scaling issues
        combined_pe = pe_temporal_expanded + pe_spatial_expanded.mean(dim=2)  # Average over channels
        
        return x_embedded + combined_pe


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for EEG data processing.
    
    Architecture:
    1. Linear projection of EEG patches to embedding dimension
    2. Positional encoding (temporal + spatial)
    3. Multi-head self-attention layers
    4. Layer normalization and feed-forward networks
    5. Global pooling to fixed-size representation
    """
    
    def __init__(
        self, 
        n_channels: int = 128,
        n_times: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        patch_size: int = 16
    ):
        """
        Initialize Transformer encoder for EEG data.
        
        Args:
            n_channels: Number of EEG channels (128 for EEG Foundation Challenge)
            n_times: Number of time samples per epoch
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            patch_size: Size of temporal patches for tokenization
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = n_times // patch_size
        
        # Input projection: Convert EEG patches to embeddings
        # Each patch is (n_channels * patch_size) -> d_model
        self.patch_embedding = nn.Linear(n_channels * patch_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, 
            max_len=self.n_patches, 
            n_channels=n_channels
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Standard FF dimension
            dropout=dropout,
            activation='gelu',  # GELU activation (better than ReLU for transformers)
            batch_first=True,
            norm_first=True     # Pre-norm architecture
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Linear(d_model, d_model)
        
        # Store hidden dimension for compatibility
        self.hidden_dim = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: Input EEG tensor of shape (batch_size, n_channels, n_times)
            
        Returns:
            Encoded representation of shape (batch_size, hidden_dim)
        """
        batch_size, n_channels, n_times = x.shape
        
        # Create temporal patches
        # Reshape to (batch_size, n_patches, n_channels, patch_size)
        x_patches = x.view(batch_size, self.n_patches, n_channels, self.patch_size)
        
        # Flatten patches for embedding: (batch_size, n_patches, n_channels * patch_size)
        x_flat = x_patches.view(batch_size, self.n_patches, -1)
        
        # Project to embedding dimension: (batch_size, n_patches, d_model)
        x_embedded = self.patch_embedding(x_flat)
        
        # Add positional encoding
        x_embedded = self.pos_encoding(x_patches, x_embedded)
        
        # Pass through transformer: (batch_size, n_patches, d_model)
        transformer_output = self.transformer(x_embedded)
        
        # Global pooling: (batch_size, d_model, 1)
        pooled = self.global_pool(transformer_output.transpose(1, 2))
        
        # Final projection: (batch_size, hidden_dim)
        output = self.final_projection(pooled.squeeze(-1))
        
        return output


class CNNEncoder(nn.Module):
    """
    CNN-based encoder for EEG data processing.
    
    Architecture:
    1. Temporal convolutions to capture time patterns
    2. Spatial convolutions to capture channel relationships
    3. Global pooling to fixed-size representation
    """
    
    def __init__(self, n_channels: int = 128, n_times: int = 256, hidden_dim: int = 512):
        """
        Initialize CNN encoder for EEG data.
        
        Args:
            n_channels: Number of EEG channels (128 for EEG Foundation Challenge)
            n_times: Number of time samples per epoch
            hidden_dim: Output hidden dimension
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.hidden_dim = hidden_dim
        
        # Temporal convolutions (1D convs along time dimension)
        self.temporal_conv1 = nn.Conv1d(n_channels, 64, kernel_size=25, padding=12)
        self.temporal_conv2 = nn.Conv1d(64, 128, kernel_size=13, padding=6)
        self.temporal_conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        
        # Spatial convolutions (1D convs along channel dimension)
        self.spatial_conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.spatial_conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Linear(256, hidden_dim)
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.
        
        Args:
            x: Input EEG tensor of shape (batch_size, n_channels, n_times)
            
        Returns:
            Encoded representation of shape (batch_size, hidden_dim)
        """
        # Temporal convolutions
        x = F.relu(self.bn1(self.temporal_conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.temporal_conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.temporal_conv3(x)))
        
        # Spatial convolutions
        x = F.relu(self.spatial_conv1(x))
        x = self.dropout(x)
        x = F.relu(self.spatial_conv2(x))
        
        # Global pooling and final projection
        pooled = self.global_pool(x)  # (batch_size, 256, 1)
        output = self.final_projection(pooled.squeeze(-1))  # (batch_size, hidden_dim)
        
        return output


class Challenge1Model(pl.LightningModule):
    """
    Challenge 1 Model: Shared Encoder + 2 Prediction Heads
    
    Multi-task learning model that predicts:
    1. Response time (regression)
    2. Hit/miss accuracy (binary classification)
    
    Optional additional features:
    - SuS EEG data can be concatenated or used as additional input
    - Demographics and psychopathology factors can be used as additional features
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Challenge 1 model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        # Model configuration
        model_config = config.get('model', {})
        encoder_type = model_config.get('encoder_type', 'transformer')
        
        # Shared encoder (CNN or Transformer)
        if encoder_type == 'transformer':
            transformer_config = model_config.get('transformer', {})
            self.shared_encoder = TransformerEncoder(
                n_channels=model_config.get('n_channels', 128),
                n_times=model_config.get('n_times', 512),
                d_model=transformer_config.get('d_model', 512),
                nhead=transformer_config.get('nhead', 8),
                num_layers=transformer_config.get('num_layers', 6),
                dropout=transformer_config.get('dropout', 0.1),
                patch_size=transformer_config.get('patch_size', 16)
            )
        else:  # CNN
            cnn_config = model_config.get('cnn', {})
            self.shared_encoder = CNNEncoder(
                n_channels=model_config.get('n_channels', 128),
                n_times=model_config.get('n_times', 512),
                hidden_dim=model_config.get('hidden_dim', 512)
            )
        
        hidden_dim = self.shared_encoder.hidden_dim
        
        # 2 prediction heads with task-specific architectures
        
        # Response time head: Specialized for behavioral timing prediction
        self.response_time_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),  # Reduced layer size for stability
            nn.ReLU(),
            nn.Dropout(0.3),  # Moderate dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Regression output
        )
        
        # Hit/miss head: Performance-focused binary classification
        self.hit_miss_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification (hit/miss)
        )
        
        # Loss weights for multi-task learning
        loss_weights = config.get('training', {}).get('loss_weights', {})
        self.loss_weights = {
            'response_time': loss_weights.get('response_time', 0.5),
            'hit_miss': loss_weights.get('hit_miss', 0.5)
        }
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Task-specific scaling factors for response time
        self.response_time_scale = config.get('training', {}).get('response_time_scale', 1000.0)
        
        # Store encoder type for logging
        self.encoder_type = encoder_type
    

    def forward(self, input_features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_features: A dictionary containing 'ccd_eeg' and optionally 'demographics'.
                            The old single-tensor format is also supported for backward compatibility.

        Returns:
            Dictionary with predictions from both heads.
        """
        # 1. Handle Input
        if isinstance(input_features, dict):
            ccd_eeg = input_features['ccd_eeg']
            demographics = input_features.get('demographics', None)
        else: # Handle old single-tensor format
            ccd_eeg = input_features
            demographics = None

        # Validate input
        if torch.isnan(ccd_eeg).any():
            raise ValueError("NaN detected in input EEG data!")
        if torch.isinf(ccd_eeg).any():
            raise ValueError("Inf detected in input EEG data!")

        # 2. Primary Feature Extraction
        # The shared encoder processes ONLY the primary CCD EEG data.
        # Output shape: (batch_size, sequence_length, hidden_dim) for Transformer
        # or (batch_size, hidden_dim) for CNN.
        shared_features = self.shared_encoder(ccd_eeg)
        
        # Validate shared features
        if torch.isnan(shared_features).any():
            raise ValueError("NaN detected in shared encoder output!")
        if torch.isinf(shared_features).any():
            raise ValueError("Inf detected in shared encoder output!")

        # 3. Global Pooling
        # To get a single feature vector per trial, we must pool the features
        # over the time/sequence dimension.
        if shared_features.dim() > 2:
            pooled_features = torch.mean(shared_features, dim=1) # Shape: (batch_size, hidden_dim)
        else:
            pooled_features = shared_features # CNN output is already pooled

        # Validate pooled features
        if torch.isnan(pooled_features).any():
            raise ValueError("NaN detected in pooled features!")
        if torch.isinf(pooled_features).any():
            raise ValueError("Inf detected in pooled features!")

        final_features = pooled_features # Use pooled features directly if no demographics

        # 4. Prediction Heads
        # The heads now operate on the final, fixed-size feature vector.
        response_time_pred = self.response_time_head(final_features)
        
        # Validate response time prediction before scaling
        if torch.isnan(response_time_pred).any():
            raise ValueError("NaN detected in response time head output!")
        if torch.isinf(response_time_pred).any():
            raise ValueError("Inf detected in response time head output!")
        
        # Ensure positive values and reasonable range (0-2000ms)
        response_time_pred = torch.clamp(response_time_pred.squeeze(-1), 0.1, 2000.0)
        
        # Apply scaling if needed
        if self.response_time_scale != 1.0:
            response_time_pred = response_time_pred * self.response_time_scale
        
        hit_miss_pred = self.hit_miss_head(final_features)
        
        # Validate hit/miss prediction
        if torch.isnan(hit_miss_pred).any():
            raise ValueError("NaN detected in hit/miss head output!")
        if torch.isinf(hit_miss_pred).any():
            raise ValueError("Inf detected in hit/miss head output!")
        
        predictions = {
            'response_time': response_time_pred, # Ensure correct output shape
            'hit_miss': hit_miss_pred
        }

        return predictions
    
    def training_step(self, batch: Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], 
                                       Tuple[Dict[str, Any], Dict[str, torch.Tensor]]], 
                     batch_idx: int) -> torch.Tensor:
        """
        Training step with multi-task learning.
        
        Args:
            batch: Tuple of (input_features_dict, targets_dict) or (eeg_data, targets_dict)
            batch_idx: Batch index
            
        Returns:
            Total training loss
        """
        # Handle both old and new batch formats
        if isinstance(batch[0], dict):
            # New format: (input_features_dict, targets)
            input_features, targets = batch
            ccd_eeg = input_features['ccd_eeg']  # Extract primary input
            optional_features = {k: v for k, v in input_features.items() if k != 'ccd_eeg'}
        else:
            # Old format: (eeg_data, targets)
            ccd_eeg, targets = batch
            optional_features = {}

        # Validate input data
        if torch.isnan(ccd_eeg).any():
            raise ValueError("NaN detected in input EEG data!")
        if torch.isinf(ccd_eeg).any():
            raise ValueError("Inf detected in input EEG data!")
        
        # Forward pass
        try:
            predictions = self(input_features)
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise e

        # Validate predictions
        if torch.isnan(predictions['response_time']).any():
            raise ValueError("NaN detected in response time predictions!")
        if torch.isnan(predictions['hit_miss']).any():
            raise ValueError("NaN detected in hit/miss predictions!")
        if torch.isinf(predictions['response_time']).any():
            raise ValueError("Inf detected in response time predictions!")
        if torch.isinf(predictions['hit_miss']).any():
            raise ValueError("Inf detected in hit/miss predictions!")
        
        # Validate targets
        if torch.isnan(targets['response_time']).any():
            raise ValueError("NaN detected in response time targets!")
        if torch.isnan(targets['hit_miss']).any():
            raise ValueError("NaN detected in hit/miss targets!")
        
        # Compute individual task losses
        losses = {}
        
        # Response time loss (regression)
        rt_pred = predictions['response_time'].squeeze()
        rt_target = targets['response_time'].squeeze()
        
        # Safety check for loss computation
        if torch.isnan(rt_pred).any() or torch.isinf(rt_pred).any():
            logger.warning("Invalid response time predictions detected in training step")
            rt_pred = torch.clamp(rt_pred, 0.1, 2000.0)
        
        if torch.isnan(rt_target).any() or torch.isinf(rt_target).any():
            logger.warning("Invalid response time targets detected in training step")
            rt_target = torch.clamp(rt_target, 0.1, 2000.0)
        
        losses['response_time'] = F.huber_loss(rt_pred, rt_target, delta=1.0)
        
        # Hit/miss loss (classification)
        hm_pred = predictions['hit_miss']
        hm_target = targets['hit_miss'].squeeze()
        losses['hit_miss'] = self.ce_loss(hm_pred, hm_target)
        
        # Validate individual losses
        for task, loss in losses.items():
            if torch.isnan(loss):
                raise ValueError(f"NaN detected in {task} loss!")
            if torch.isinf(loss):
                raise ValueError(f"Inf detected in {task} loss!")
        
        # Weighted multi-task loss
        total_loss = sum(self.loss_weights[task] * loss for task, loss in losses.items())

        if torch.isnan(total_loss):
            raise ValueError("Total loss became NaN!")
        if torch.isinf(total_loss):
            raise ValueError("Total loss became Inf!")
        
        # Log individual losses
        for task, loss in losses.items():
            self.log(f'train/{task}_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        
        # Log total loss
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], 
                                         Tuple[Dict[str, Any], Dict[str, torch.Tensor]]], 
                       batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with comprehensive metrics.
        
        Args:
            batch: Tuple of (input_features_dict, targets_dict) or (eeg_data, targets_dict)
            batch_idx: Batch index
            
        Returns:
            Dictionary of validation metrics
        """
        # Handle both old and new batch formats
        if isinstance(batch[0], dict):
            # New format: (input_features_dict, targets)
            input_features, targets = batch
            ccd_eeg = input_features['ccd_eeg']  # Extract primary input
            optional_features = {k: v for k, v in input_features.items() if k != 'ccd_eeg'}
        else:
            # Old format: (eeg_data, targets)
            ccd_eeg, targets = batch
            optional_features = {}
        
        # Forward pass
        predictions = self(input_features)
        
        # Compute individual task losses
        losses = {}
        
        # Response time loss (regression)
        rt_pred = predictions['response_time'].squeeze()
        rt_target = targets['response_time'].squeeze()
        
        # Safety check for loss computation
        if torch.isnan(rt_pred).any() or torch.isinf(rt_pred).any():
            logger.warning("Invalid response time predictions detected in validation step")
            rt_pred = torch.clamp(rt_pred, 0.1, 2000.0)
        
        if torch.isnan(rt_target).any() or torch.isinf(rt_target).any():
            logger.warning("Invalid response time targets detected in validation step")
            rt_target = torch.clamp(rt_target, 0.1, 2000.0)
        
        losses['response_time'] = F.huber_loss(rt_pred, rt_target, delta=1.0)
        
        # Hit/miss loss (classification)
        losses['hit_miss'] = self.ce_loss(predictions['hit_miss'], targets['hit_miss'].squeeze())
        
        # Weighted multi-task loss
        total_loss = sum(self.loss_weights[task] * loss for task, loss in losses.items())
        
        # Compute additional metrics
        metrics = {}

        # Regression metrics (R², MAE, RMSE)
        rt_targets = targets['response_time'].cpu().numpy()
        rt_predictions = predictions['response_time'].squeeze().cpu().numpy()
        
        # Apply numerical stability checks
        if np.any(np.isnan(rt_predictions)) or np.any(np.isinf(rt_predictions)):
            logger.warning(f"Invalid predictions detected: NaN={np.any(np.isnan(rt_predictions))}, Inf={np.any(np.isinf(rt_predictions))}")
            # Replace invalid values with median of targets
            rt_predictions = np.where(np.isfinite(rt_predictions), rt_predictions, np.median(rt_targets))
        
        if np.any(np.isnan(rt_targets)) or np.any(np.isinf(rt_targets)):
            logger.warning(f"Invalid targets detected: NaN={np.any(np.isnan(rt_targets))}, Inf={np.any(np.isinf(rt_targets))}")
            # Replace invalid values with median
            rt_targets = np.where(np.isfinite(rt_targets), rt_targets, np.median(rt_targets))
        
        # Ensure both arrays have finite values before computing metrics
        if np.all(np.isfinite(rt_targets)) and np.all(np.isfinite(rt_predictions)):
            metrics['response_time_r2'] = r2_score(rt_targets, rt_predictions)
            metrics['response_time_mae'] = mean_absolute_error(rt_targets, rt_predictions)
            metrics['response_time_rmse'] = np.sqrt(mean_squared_error(rt_targets, rt_predictions))
        else:
            logger.warning("Cannot compute metrics due to invalid values")
            metrics['response_time_r2'] = 0.0
            metrics['response_time_mae'] = float('inf')
            metrics['response_time_rmse'] = float('inf')

        # Classification metrics (F1)
        hm_targets = targets['hit_miss'].cpu().numpy()
        hm_pred_class = torch.argmax(predictions['hit_miss'], dim=1).cpu().numpy()
        metrics['hit_miss_f1'] = f1_score(hm_targets, hm_pred_class, average='weighted')

        # Log all metrics
        for task, loss in losses.items():
            self.log(f'val/{task}_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_step=False, on_epoch=True, prog_bar=False)
        
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': total_loss,
            'val_metrics': metrics
        }
    
    def test_step(self, batch: Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], 
                                    Tuple[Dict[str, Any], Dict[str, torch.Tensor]]], 
                  batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step with comprehensive metrics.
        
        Args:
            batch: Tuple of (input_features_dict, targets_dict) or (eeg_data, targets_dict)
            batch_idx: Batch index
            
        Returns:
            Dictionary of test metrics
        """
        # Handle both old and new batch formats
        if isinstance(batch[0], dict):
            # New format: (input_features_dict, targets)
            input_features, targets = batch
            ccd_eeg = input_features['ccd_eeg']  # Extract primary input
            optional_features = {k: v for k, v in input_features.items() if k != 'ccd_eeg'}
        else:
            # Old format: (eeg_data, targets)
            ccd_eeg, targets = batch
            optional_features = {}
        
        # Forward pass
        predictions = self(input_features)
        
        # Compute individual task losses
        losses = {}
        
        # Response time loss (regression)
        losses['response_time'] = self.mse_loss(predictions['response_time'].squeeze(), targets['response_time'].squeeze())
        
        # Hit/miss loss (classification)
        losses['hit_miss'] = self.ce_loss(predictions['hit_miss'], targets['hit_miss'].squeeze())
        
        # Weighted multi-task loss
        total_loss = sum(self.loss_weights[task] * loss for task, loss in losses.items())
        
        # Compute additional metrics
        metrics = {}
        
        # Response time loss (regression) - safety check for test step
        rt_pred = predictions['response_time'].squeeze()
        rt_target = targets['response_time'].squeeze()
        
        # Safety check for loss computation
        if torch.isnan(rt_pred).any() or torch.isinf(rt_pred).any():
            logger.warning("Invalid response time predictions detected in test step")
            rt_pred = torch.clamp(rt_pred, -1e3, 1e3)
        
        if torch.isnan(rt_target).any() or torch.isinf(rt_target).any():
            logger.warning("Invalid response time targets detected in test step")
            rt_target = torch.clamp(rt_target, -1e3, 1e3)
        
        # Regression metrics (R², MAE, RMSE)
        rt_targets = targets['response_time'].cpu().numpy()
        rt_predictions = predictions['response_time'].squeeze().cpu().numpy()
        
        # Apply numerical stability checks
        if np.any(np.isnan(rt_predictions)) or np.any(np.isinf(rt_predictions)):
            logger.warning(f"Invalid predictions detected: NaN={np.any(np.isnan(rt_predictions))}, Inf={np.any(np.isinf(rt_predictions))}")
            # Replace invalid values with median of targets
            rt_predictions = np.where(np.isfinite(rt_predictions), rt_predictions, np.median(rt_targets))
        
        if np.any(np.isnan(rt_targets)) or np.any(np.isinf(rt_targets)):
            logger.warning(f"Invalid targets detected: NaN={np.any(np.isnan(rt_targets))}, Inf={np.any(np.isinf(rt_targets))}")
            # Replace invalid values with median
            rt_targets = np.where(np.isfinite(rt_targets), rt_targets, np.median(rt_targets))
        
        # Ensure both arrays have finite values before computing metrics
        if np.all(np.isfinite(rt_targets)) and np.all(np.isfinite(rt_predictions)):
            metrics['response_time_r2'] = r2_score(rt_targets, rt_predictions)
            metrics['response_time_mae'] = mean_absolute_error(rt_targets, rt_predictions)
            metrics['response_time_rmse'] = np.sqrt(mean_squared_error(rt_targets, rt_predictions))
        else:
            logger.warning("Cannot compute metrics due to invalid values")
            metrics['response_time_r2'] = 0.0
            metrics['response_time_mae'] = float('inf')
            metrics['response_time_rmse'] = float('inf')
        
        # Classification metrics (F1)
        hm_targets = targets['hit_miss'].cpu().numpy()
        hm_pred_class = torch.argmax(predictions['hit_miss'], dim=1).cpu().numpy()
        metrics['hit_miss_f1'] = f1_score(hm_targets, hm_pred_class, average='weighted')
        
        # Log all metrics
        for task, loss in losses.items():
            self.log(f'test/{task}_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        
        for metric_name, metric_value in metrics.items():
            self.log(f'test/{metric_name}', metric_value, on_step=False, on_epoch=True, prog_bar=False)
        
        self.log('test/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'test_loss': total_loss,
            'test_metrics': metrics
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with NaN prevention."""
        training_config = self.hparams.get('training', {})
        
        # Use lower learning rate for transformers to prevent NaN
        base_lr = training_config.get('learning_rate', 1e-3)
        if self.encoder_type == 'transformer':
            base_lr = min(base_lr, 1e-4)  # Further reduce transformer learning rate
        
        # Optimizer with better numerical stability
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,
            weight_decay=training_config.get('weight_decay', 1e-4),
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup for transformers
        scheduler_config = training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if self.encoder_type == 'transformer' and scheduler_type == 'cosine':
            # Use warmup + cosine annealing for transformers
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            warmup_start_lr = scheduler_config.get('warmup_start_lr', 1e-5)
            
            # Create warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_lr / base_lr,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            # Create main scheduler
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - warmup_epochs,
                eta_min=training_config.get('min_lr', 1e-6)
            )
            
            # Combine warmup and main scheduler
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss'
            }
        }

def main():
    """
    Main function to train Challenge 1 model.
    """
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'challenge1_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # GPU detection and configuration
    gpus = detect_gpus()
    gpu_config = get_recommended_config(gpus)
    
    print("="*50)
    print("Challenge 1: Cross-Task Transfer Learning")
    print("="*50)
    print(f"Training with CCD EEG data as primary input (X1)")
    print(f"Predicting CCD behavioral outcomes (response time, hit/miss)")
    
    # Test both encoder types
    for encoder_type in ['cnn', 'transformer']:
        print(f"\n  Testing {encoder_type.upper()} Encoder:")
        
        # Update config for encoder type
        test_config = config.copy()
        test_config['model']['encoder_type'] = encoder_type
        
        # Initialize model
        model = Challenge1Model(test_config)
        
        # Print model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   • Encoder Type: {encoder_type.upper()}")
        print(f"   • Shared Encoder: {model.shared_encoder.hidden_dim}D features")
        print(f"   • Total Parameters: {total_params:,}")
        print(f"   • Trainable Parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 129, 512)  # Batch of 4 EEG samples (2s at 256Hz)
        with torch.no_grad():
            predictions = model(dummy_input)
            print(f"   • Forward pass successful!")
            print(f"   • Output shapes: {[f'{k}:{v.shape}' for k, v in predictions.items()]}")
    
    print("="*50)
    print("Both encoder architectures ready for training!")

if __name__ == '__main__':
    main() 