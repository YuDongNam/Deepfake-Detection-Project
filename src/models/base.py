"""
Base classes and common utilities for deepfake detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss is designed to address the one-stage object detection scenario where
    there is an extreme imbalance between foreground and background classes.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Computed focal loss
        """
        BCE = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal_loss.mean()


class BaseDeepfakeModel(nn.Module):
    """
    Base class for deepfake detection models.
    
    This class provides common functionality and interface for all deepfake detection models.
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_name': self.__class__.__name__,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        raise NotImplementedError("Subclasses must implement forward method")


def create_model(model_type: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function to create model instances.
    
    Args:
        model_type (str): Type of model to create ("swin" or "xception")
        num_classes (int): Number of output classes
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == "swin":
        from .swin_transformer import VideoSwinTransformer
        return VideoSwinTransformer(num_classes=num_classes, **kwargs)
    elif model_type == "xception":
        from .xception_lstm import XceptionLSTM
        return XceptionLSTM(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'swin' or 'xception'.")


def load_model(model_path: str, model_type: str, device: torch.device, 
                num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model to load ("swin" or "xception")
        device (torch.device): Device to load the model on
        num_classes (int): Number of output classes
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        Loaded model instance
    """
    model = create_model(model_type, num_classes=num_classes, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
