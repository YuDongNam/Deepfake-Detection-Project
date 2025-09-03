"""
Training module for deepfake detection models.

This module contains the training loop, validation, and early stopping functionality
for both Video Swin Transformer and XceptionNet+LSTM models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, average_precision_score
)

from .models import FocalLoss, create_model
from .evaluate import evaluate_model


class TrainingHistory:
    """Class to track training history and metrics."""
    
    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': [],
            'train_prc': [], 'val_prc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_mcc': [], 'val_mcc': []
        }
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update history with new metrics."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_auc') -> int:
        """Get the epoch with the best metric value."""
        if metric not in self.history:
            raise ValueError(f"Metric {metric} not found in history")
        
        if 'loss' in metric:
            return np.argmin(self.history[metric])
        else:
            return np.argmax(self.history[metric])
    
    def get_best_value(self, metric: str = 'val_auc') -> float:
        """Get the best value for a metric."""
        if metric not in self.history:
            raise ValueError(f"Metric {metric} not found in history")
        
        if 'loss' in metric:
            return np.min(self.history[metric])
        else:
            return np.max(self.history[metric])


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, 
                threshold: float = 0.4) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    train_loss = []
    preds, targets, probs = [], [], []
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        if len(batch) == 4:  # Video Swin Transformer format
            video_tensor, label, video_id, face_idx = batch
            video_tensor = video_tensor.to(device)
        else:  # XceptionNet+LSTM format
            frames, label, video_id, face_idx = batch
            video_tensor = frames.to(device)
        
        label = label.to(device)
        
        optimizer.zero_grad()
        out = model(video_tensor)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        
        # Calculate predictions and probabilities
        prob = torch.softmax(out, dim=-1)[:, 1].detach().cpu().numpy()
        pred_label = (prob > threshold).astype(int)
        
        preds.extend(pred_label)
        targets.extend(label.cpu().numpy())
        probs.extend(prob)
    
    # Calculate metrics
    train_loss = np.mean(train_loss)
    train_acc = accuracy_score(targets, preds)
    train_f1 = f1_score(targets, preds)
    train_auc = roc_auc_score(targets, probs)
    train_prc = average_precision_score(targets, probs)
    train_prec = precision_score(targets, preds)
    train_rec = recall_score(targets, preds)
    train_mcc = matthews_corrcoef(targets, preds)
    
    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_prc': train_prc,
        'train_prec': train_prec,
        'train_rec': train_rec,
        'train_mcc': train_mcc
    }


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device, threshold: float = 0.4) -> Dict[str, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    val_loss = []
    preds, targets, probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if len(batch) == 4:  # Video Swin Transformer format
                video_tensor, label, video_id, face_idx = batch
                video_tensor = video_tensor.to(device)
            else:  # XceptionNet+LSTM format
                frames, label, video_id, face_idx = batch
                video_tensor = frames.to(device)
            
            label = label.to(device)
            
            out = model(video_tensor)
            loss = criterion(out, label)
            
            val_loss.append(loss.item())
            
            # Calculate predictions and probabilities
            prob = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            pred_label = (prob > threshold).astype(int)
            
            preds.extend(pred_label)
            targets.extend(label.cpu().numpy())
            probs.extend(prob)
    
    # Calculate metrics
    val_loss = np.mean(val_loss)
    val_acc = accuracy_score(targets, preds)
    val_f1 = f1_score(targets, preds)
    val_auc = roc_auc_score(targets, probs)
    val_prc = average_precision_score(targets, probs)
    val_prec = precision_score(targets, preds)
    val_rec = recall_score(targets, preds)
    val_mcc = matthews_corrcoef(targets, preds)
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_prc': val_prc,
        'val_prec': val_prec,
        'val_rec': val_rec,
        'val_mcc': val_mcc
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 30, learning_rate: float = 1e-4, 
                patience: int = 10, save_path: str = "best_model.pth",
                device: torch.device = None, threshold: float = 0.4,
                use_focal_loss: bool = True, focal_alpha: float = 0.1,
                focal_gamma: float = 2.0) -> Tuple[nn.Module, TrainingHistory]:
    """
    Train a deepfake detection model with early stopping.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait before early stopping
        save_path: Path to save the best model
        device: Device to run training on
        threshold: Threshold for binary classification
        use_focal_loss: Whether to use Focal Loss
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
        
    Returns:
        Tuple of (best_model, training_history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Setup loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize training history
    history = TrainingHistory()
    
    # Early stopping variables
    best_auc = 0
    patience_counter = 0
    
    print(f"Starting training on device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, threshold)
        
        # Validation phase
        val_metrics = validate_epoch(model, val_loader, criterion, device, threshold)
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        history.update(epoch_metrics)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['train_loss']:.4f}, "
              f"Acc: {train_metrics['train_acc']:.4f}, "
              f"F1: {train_metrics['train_f1']:.4f}, "
              f"AUC: {train_metrics['train_auc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
              f"Acc: {val_metrics['val_acc']:.4f}, "
              f"F1: {val_metrics['val_f1']:.4f}, "
              f"AUC: {val_metrics['val_auc']:.4f}")
        
        # Early stopping and model saving
        if val_metrics['val_auc'] > best_auc:
            best_auc = val_metrics['val_auc']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved! Val AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        print("-" * 50)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    print(f"Training completed! Best validation AUC: {best_auc:.4f}")
    
    return model, history


def train_with_config(model_type: str, train_loader: DataLoader, val_loader: DataLoader,
                     config: dict) -> Tuple[nn.Module, TrainingHistory]:
    """
    Train a model using configuration dictionary.
    
    Args:
        model_type: Type of model ("swin" or "xception")
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary containing training parameters
        
    Returns:
        Tuple of (best_model, training_history)
    """
    # Create model
    model = create_model(
        model_type=model_type,
        num_classes=config.get('num_classes', 2),
        **config.get('model_kwargs', {})
    )
    
    # Train model
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 30),
        learning_rate=config.get('learning_rate', 1e-4),
        patience=config.get('patience', 10),
        save_path=config.get('save_path', f'best_{model_type}_model.pth'),
        device=config.get('device'),
        threshold=config.get('threshold', 0.4),
        use_focal_loss=config.get('use_focal_loss', True),
        focal_alpha=config.get('focal_alpha', 0.1),
        focal_gamma=config.get('focal_gamma', 2.0)
    )
