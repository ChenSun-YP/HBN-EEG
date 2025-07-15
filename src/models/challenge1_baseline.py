#!/usr/bin/env python3
"""
Challenge 1: Cross-Task Transfer Learning - Shared Encoder + 4 Heads
EEG Foundation Challenge 2025

OFFICIAL CHALLENGE CONSTRAINTS:
==============================
- Training Input: ONLY SuS (Surround Suppression) EEG epochs
- Prediction Targets: CCD behavioral outcomes per trial:
  * Response time (regression)
  * Hit/miss accuracy (binary classification) 
  * Age (auxiliary regression)
  * Sex (auxiliary classification)
- Constraint: CCD EEG data is NOT provided and NOT allowed for training
- Architecture: Shared encoder with 4 prediction heads

Model Architecture:
==================
                    ┌─ Age Head (regression)
                    ├─ Sex Head (classification)
SuS EEG → Encoder → ├─ Response Time Head (regression)
                    └─ Hit/Miss Head (binary classification)

Encoder Options:
- CNN-based: Temporal + Spatial convolutions (original)
- Transformer-based: Multi-head self-attention with positional encoding

The shared encoder learns representations from SuS EEG data that can predict
behavioral outcomes on the CCD task, along with demographic information.

Training Strategy:
=================
- Multi-task learning with weighted loss functions
- Shared encoder learns task-invariant representations
- Task-specific heads capture different aspects of cognition
- Auxiliary tasks (age, sex) provide regularization

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
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

import mne
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from utils.gpu_utils import detect_gpus, get_recommended_config, print_gpu_memory_summary


class PositionalEncoding(nn.Module):
    """
    Positional encoding for EEG data with both temporal and spatial dimensions.
    
    For EEG data (n_channels, n_times), we need to encode:
    1. Temporal position: where in time each sample occurs
    2. Spatial position: which channel each sample comes from
    """
    
    def __init__(self, d_model: int, max_len: int = 512, n_channels: int = 129):
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
        n_channels: int = 129, 
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
            n_channels: Number of EEG channels (129 for HBN)
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
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 because we concat avg and max pooling
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.hidden_dim = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: Input EEG tensor of shape (batch_size, n_channels, n_times)
            
        Returns:
            Encoded features of shape (batch_size, d_model)
        """
        batch_size, n_channels, n_times = x.shape
        
        # Create patches: (batch_size, n_channels, n_times) -> (batch_size, n_patches, n_channels, patch_size)
        x_patches = x.unfold(2, self.patch_size, self.patch_size)  # (batch_size, n_channels, n_patches, patch_size)
        x_patches = x_patches.permute(0, 2, 1, 3)  # (batch_size, n_patches, n_channels, patch_size)
        
        # Keep spatial structure for positional encoding, but also create flattened version for embedding
        x_patches_flattened = x_patches.reshape(batch_size, self.n_patches, -1)  # (batch_size, n_patches, n_channels * patch_size)
        
        # Linear projection to embedding space
        x_embedded = self.patch_embedding(x_patches_flattened)  # (batch_size, n_patches, d_model)
        
        # Add 2D positional encoding (spatial + temporal)
        x_encoded = self.pos_encoding(x_patches, x_embedded)
        
        # Apply transformer layers
        transformer_out = self.transformer(x_encoded)  # (batch_size, n_patches, d_model)
        
        # Layer normalization
        transformer_out = self.layer_norm(transformer_out)
        
        # Global pooling: combine average and max pooling
        # Transpose for pooling: (batch_size, n_patches, d_model) -> (batch_size, d_model, n_patches)
        transformer_out_t = transformer_out.transpose(1, 2)
        
        avg_pooled = self.global_avg_pool(transformer_out_t).squeeze(-1)  # (batch_size, d_model)
        max_pooled = self.global_max_pool(transformer_out_t).squeeze(-1)  # (batch_size, d_model)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # (batch_size, d_model * 2)
        
        # Final projection
        output = self.output_projection(pooled)  # (batch_size, d_model)
        
        return output


class CNNEncoder(nn.Module):
    """
    CNN-based shared encoder for Challenge 1 that processes SuS EEG data
    and learns representations predictive of CCD behavioral outcomes.
    
    Architecture:
    - Convolutional layers for temporal-spatial feature extraction
    - Batch normalization and dropout for regularization
    - Global average pooling for sequence-to-vector conversion
    - Fully connected layers for high-level representation learning
    """
    
    def __init__(self, n_channels: int = 129, n_times: int = 256, hidden_dim: int = 512):
        """
        Initialize the shared CNN encoder.
        
        Args:
            n_channels: Number of EEG channels (129 for HBN)
            n_times: Number of time samples per epoch
            hidden_dim: Hidden dimension for the final representation
        """
        super().__init__()
        
        # Temporal convolution: Extract temporal patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),  # 25 samples = ~100ms at 256Hz
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=13, padding=6),  # 13 samples = ~50ms at 256Hz
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Spatial convolution: Extract spatial patterns across channels
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),  # 7 samples = ~25ms at 256Hz
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),  # 3 samples = ~12ms at 256Hz
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Global average pooling to convert sequences to vectors
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for high-level representation
        self.fc_layers = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shared CNN encoder.
        
        Args:
            x: Input EEG tensor of shape (batch_size, n_channels, n_times)
            
        Returns:
            Encoded features of shape (batch_size, hidden_dim)
        """
        # Temporal convolution
        x = self.temporal_conv(x)  # (batch_size, 128, n_times)
        
        # Spatial convolution
        x = self.spatial_conv(x)   # (batch_size, 256, n_times)
        
        # Global average pooling
        x = self.global_pool(x)    # (batch_size, 256, 1)
        x = x.squeeze(-1)          # (batch_size, 256)
        
        # Fully connected layers
        x = self.fc_layers(x)      # (batch_size, hidden_dim)
        
        return x


class Challenge1Model(pl.LightningModule):
    """
    Challenge 1 model with shared encoder and 4 prediction heads.
    
    This model implements the official Challenge 1 architecture:
    - Shared encoder processes SuS EEG data (CNN or Transformer)
    - 4 prediction heads for different targets:
      * Age head: Regression for age prediction
      * Sex head: Binary classification for sex prediction
      * Response time head: Regression for CCD response time
      * Hit/miss head: Binary classification for CCD accuracy
    
    Training uses multi-task learning with weighted loss functions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Challenge 1 model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Choose encoder type
        encoder_type = config.get('encoder_type', 'cnn')  # 'cnn' or 'transformer'
        
        if encoder_type == 'transformer':
            # Transformer-based encoder
            self.shared_encoder = TransformerEncoder(
                n_channels=config.get('n_channels', 129),
                n_times=config.get('n_times', 256),
                d_model=config.get('hidden_dim', 512),
                nhead=config.get('transformer_heads', 8),
                num_layers=config.get('transformer_layers', 6),
                dropout=config.get('dropout', 0.1),
                patch_size=config.get('patch_size', 16)
            )
        else:
            # CNN-based encoder (default)
            self.shared_encoder = CNNEncoder(
                n_channels=config.get('n_channels', 129),
                n_times=config.get('n_times', 256),
                hidden_dim=config.get('hidden_dim', 512)
            )
        
        hidden_dim = self.shared_encoder.hidden_dim
        
        # 4 prediction heads with task-specific architectures
        
        # Age head: Deeper regression with layer normalization (demographic prediction)
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression output
        )
        
        # Sex head: Focused binary classification with specialized activation
        self.sex_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Lower dropout for binary classification
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification (male/female)
        )
        
        # Response time head: Specialized for behavioral timing prediction
        self.response_time_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),  # Larger first layer for timing
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout for robustness
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # Regression output
            nn.Sigmoid()  # Constrain to [0,1] then scale in loss
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
            'age': loss_weights.get('age', 0.1),
            'sex': loss_weights.get('sex', 0.1),
            'response_time': loss_weights.get('response_time', 0.4),
            'hit_miss': loss_weights.get('hit_miss', 0.4)
        }
        
        # Task-specific auxiliary loss weights (based on literature)
        aux_loss_weights = config.get('training', {}).get('aux_loss_weights', {})
        self.aux_loss_weights = {
            'sparsity': aux_loss_weights.get('sparsity', 0.01),        # Lachapelle et al. (2023)
            'task_vector': aux_loss_weights.get('task_vector', 0.005),  # Cheng et al. (2025)
            'gradient_conflict': aux_loss_weights.get('gradient_conflict', 0.002)  # Yu et al. (2020)
        }
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Task-specific scaling factors for response time
        self.response_time_scale = config.get('training', {}).get('response_time_scale', 1000.0)
        
        # Store encoder type for logging
        self.encoder_type = encoder_type
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input SuS EEG tensor of shape (batch_size, n_channels, n_times)
            
        Returns:
            Dictionary with predictions from all 4 heads and intermediate features
        """
        # Shared encoder
        shared_features = self.shared_encoder(x)
        
        # 4 prediction heads with task-specific processing
        age_pred = self.age_head(shared_features)
        sex_pred = self.sex_head(shared_features)
        response_time_pred = self.response_time_head(shared_features) * self.response_time_scale
        hit_miss_pred = self.hit_miss_head(shared_features)
        
        predictions = {
            'age': age_pred,
            'sex': sex_pred,
            'response_time': response_time_pred,
            'hit_miss': hit_miss_pred
        }
        
        # Store information for auxiliary losses (only during training)
        if self.training:
            predictions['shared_features'] = shared_features
            predictions['head_weights'] = {
                'age': self._get_head_weights(self.age_head),
                'sex': self._get_head_weights(self.sex_head),
                'response_time': self._get_head_weights(self.response_time_head),
                'hit_miss': self._get_head_weights(self.hit_miss_head)
            }
        
        return predictions
    
    def _get_head_weights(self, head: nn.Sequential) -> torch.Tensor:
        """Extract first layer weights from a head for regularization."""
        for layer in head:
            if isinstance(layer, nn.Linear):
                return layer.weight
        return None
    
    def _compute_sparsity_loss(self, head_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute sparsity loss to encourage sparse task-specific predictors.
        
        Based on Lachapelle et al. (2023): "Synergies between Disentanglement and Sparsity: 
        Generalization and Identifiability in Multi-Task Learning"
        
        Sparse task-specific predictors help with disentanglement and generalization.
        """
        total_loss = 0.0
        count = 0
        
        for task_name, weights in head_weights.items():
            if weights is not None:
                # L1 regularization on first layer weights
                l1_loss = torch.norm(weights, p=1)
                total_loss += l1_loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=list(head_weights.values())[0].device)
    
    def _compute_task_vector_loss(self, head_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute task vector loss to encourage different learning directions.
        
        Based on task vector concepts from model merging literature.
        Encourages each head to learn in different directions from shared representation.
        Only compares heads of the same size to avoid dimension mismatch.
        """
        weights = [w for w in head_weights.values() if w is not None]
        if len(weights) < 2:
            return torch.tensor(0.0, device=weights[0].device)
        
        total_loss = 0.0
        count = 0
        
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                # Flatten weights for comparison
                w1_flat = weights[i].flatten()
                w2_flat = weights[j].flatten()
                
                # Only compare heads of the same size to avoid dimension mismatch
                if w1_flat.size() != w2_flat.size():
                    continue
                
                # Normalize to unit vectors
                w1_norm = torch.nn.functional.normalize(w1_flat, dim=0)
                w2_norm = torch.nn.functional.normalize(w2_flat, dim=0)
                
                # Encourage different directions (minimize dot product)
                similarity = torch.dot(w1_norm, w2_norm)
                total_loss += similarity.abs()  # Penalize both positive and negative correlation
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=weights[0].device)
    
    def _compute_gradient_conflict_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute gradient conflict loss to detect and mitigate conflicting gradients.
        
        Based on Yu et al. (2020) and related work on gradient conflicts in multi-task learning.
        When gradients from different tasks conflict, training becomes unstable.
        
        Note: Using create_graph=False for efficiency since we don't need second-order derivatives.
        """
        if len(losses) < 2:
            return torch.tensor(0.0, device=list(losses.values())[0].device)
        
        # Get gradients with respect to shared encoder parameters
        shared_params = list(self.shared_encoder.parameters())
        if not shared_params:
            return torch.tensor(0.0, device=list(losses.values())[0].device)
        
        gradients = {}
        for task_name, loss in losses.items():
            # Use create_graph=False for efficiency - we don't need second-order derivatives
            grad = torch.autograd.grad(
                loss, shared_params, retain_graph=True, create_graph=False
            )
            # Concatenate all gradients into a single vector
            gradients[task_name] = torch.cat([g.flatten() for g in grad])
        
        # Compute pairwise cosine similarities between gradients
        grad_list = list(gradients.values())
        total_conflict = 0.0
        count = 0
        
        for i in range(len(grad_list)):
            for j in range(i + 1, len(grad_list)):
                # Normalize gradients
                grad_i = torch.nn.functional.normalize(grad_list[i], dim=0)
                grad_j = torch.nn.functional.normalize(grad_list[j], dim=0)
                
                # Compute cosine similarity
                similarity = torch.dot(grad_i, grad_j)
                
                # Penalize negative similarity (conflicting gradients)
                conflict = torch.clamp(-similarity, min=0.0)
                total_conflict += conflict
                count += 1
        
        return total_conflict / count if count > 0 else torch.tensor(0.0, device=grad_list[0].device)

    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """
        Training step with multi-task loss.
        
        Args:
            batch: Tuple of (SuS EEG data, targets dictionary)
            batch_idx: Batch index
            
        Returns:
            Total weighted loss
        """
        x, targets = batch
        predictions = self(x)
        
        # Calculate individual losses
        losses = {}
        
        # Age loss (regression)
        if 'age' in targets:
            losses['age'] = self.mse_loss(predictions['age'].squeeze(), targets['age'].float().squeeze())
        
        # Sex loss (classification)
        if 'sex' in targets:
            losses['sex'] = self.ce_loss(predictions['sex'], targets['sex'].long().squeeze())
        
        # Response time loss (regression) - normalized by scale
        if 'response_time' in targets:
            losses['response_time'] = self.mse_loss(
                predictions['response_time'].squeeze() / self.response_time_scale, 
                targets['response_time'].float().squeeze() / self.response_time_scale
            )
        
        # Hit/miss loss (classification)
        if 'hit_miss' in targets:
            losses['hit_miss'] = self.ce_loss(predictions['hit_miss'], targets['hit_miss'].long().squeeze())
        
        # Main task weighted loss
        main_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
        
        # Auxiliary losses for task-specific learning (literature-based)
        aux_losses = {}
        if 'head_weights' in predictions:
            aux_losses['sparsity'] = self._compute_sparsity_loss(predictions['head_weights'])
            aux_losses['task_vector'] = self._compute_task_vector_loss(predictions['head_weights'])
            aux_losses['gradient_conflict'] = self._compute_gradient_conflict_loss(losses)
        
        # Total loss with auxiliary terms
        aux_loss = sum(self.aux_loss_weights[k] * v for k, v in aux_losses.items())
        total_loss = main_loss + aux_loss
        
        # Log individual losses
        for k, v in losses.items():
            self.log(f'train/{k}_loss', v, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log auxiliary losses
        for k, v in aux_losses.items():
            self.log(f'train/aux_{k}_loss', v, on_step=True, on_epoch=True, prog_bar=False)
        
        self.log('train/main_loss', main_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with metrics calculation.
        
        Note: Auxiliary losses (sparsity, task_vector, gradient_conflict) are intentionally
        excluded from validation as they are training-only regularization techniques.
        
        Args:
            batch: Tuple of (SuS EEG data, targets dictionary)
            batch_idx: Batch index
            
        Returns:
            Dictionary with validation metrics
        """
        x, targets = batch
        predictions = self(x)
        
        # Calculate losses
        losses = {}
        metrics = {}
        
        # Age metrics
        if 'age' in targets:
            age_loss = self.mse_loss(predictions['age'].squeeze(), targets['age'].float().squeeze())
            losses['age'] = age_loss
            
            # R² score for age
            age_pred = predictions['age'].squeeze().cpu().numpy()
            age_true = targets['age'].squeeze().cpu().numpy()
            if len(age_pred) > 1:  # Need at least 2 samples for R²
                r2 = r2_score(age_true, age_pred)
                metrics['age_r2'] = r2
        
        # Sex metrics
        if 'sex' in targets:
            sex_loss = self.ce_loss(predictions['sex'], targets['sex'].long().squeeze())
            losses['sex'] = sex_loss
            
            # Accuracy for sex
            sex_pred = torch.argmax(predictions['sex'], dim=1)
            sex_acc = accuracy_score(targets['sex'].squeeze().cpu().numpy(), sex_pred.cpu().numpy())
            metrics['sex_acc'] = sex_acc
        
        # Response time metrics - apply same scaling as training for consistency
        if 'response_time' in targets:
            rt_loss = self.mse_loss(
                predictions['response_time'].squeeze() / self.response_time_scale,
                targets['response_time'].float().squeeze() / self.response_time_scale
            )
            losses['response_time'] = rt_loss
            
            # R² score for response time (using original scale for interpretability)
            rt_pred = predictions['response_time'].squeeze().cpu().numpy()
            rt_true = targets['response_time'].squeeze().cpu().numpy()
            if len(rt_pred) > 1:
                r2 = r2_score(rt_true, rt_pred)
                metrics['response_time_r2'] = r2
        
        # Hit/miss metrics
        if 'hit_miss' in targets:
            hm_loss = self.ce_loss(predictions['hit_miss'], targets['hit_miss'].long().squeeze())
            losses['hit_miss'] = hm_loss
            
            # Accuracy for hit/miss
            hm_pred = torch.argmax(predictions['hit_miss'], dim=1)
            hm_acc = accuracy_score(targets['hit_miss'].squeeze().cpu().numpy(), hm_pred.cpu().numpy())
            metrics['hit_miss_acc'] = hm_acc
        
        # Total loss
        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
        
        # Log validation metrics
        for k, v in losses.items():
            self.log(f'val/{k}_loss', v, on_step=False, on_epoch=True, prog_bar=True)
        
        for k, v in metrics.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': total_loss, **losses, **metrics}
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('max_epochs', 100),
            eta_min=self.config.get('min_lr', 1e-6)
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
    print(f"Training with ONLY SuS EEG data")
    print(f"Predicting CCD behavioral outcomes + demographics")
    
    # Test both encoder types
    for encoder_type in ['cnn', 'transformer']:
        print(f"\n  Testing {encoder_type.upper()} Encoder:")
        
        # Update config for encoder type
        test_config = config.copy()
        test_config['encoder_type'] = encoder_type
        
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