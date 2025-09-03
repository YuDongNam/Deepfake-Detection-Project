"""
Model architectures for deepfake detection.

This module contains the model definitions for both Video Swin Transformer and XceptionNet+LSTM architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from typing import Optional


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


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer model for deepfake detection.
    
    This model uses a pre-trained Video Swin Transformer as the backbone and
    replaces the final classification head for binary deepfake detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        pretrained (bool): Whether to use pre-trained weights (default: True)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            weights = Swin3D_T_Weights.DEFAULT
            self.backbone = swin3d_t(weights=weights)
        else:
            self.backbone = swin3d_t(weights=None)
        
        # Replace the classification head
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W) where:
               B = batch size
               C = channels (3 for RGB)
               T = temporal dimension (number of frames)
               H, W = height and width of frames
               
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        return self.backbone(x)


class XceptionLSTM(nn.Module):
    """
    XceptionNet + LSTM model for deepfake detection.
    
    This model uses XceptionNet as a feature extractor for each frame,
    followed by an LSTM to process temporal sequences.
    
    Args:
        hidden_size (int): LSTM hidden size (default: 256)
        num_layers (int): Number of LSTM layers (default: 1)
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to use pre-trained XceptionNet weights (default: True)
    """
    
    def __init__(self, hidden_size: int = 256, num_layers: int = 1, 
                 num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # Import timm for XceptionNet
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for XceptionNet. Install with: pip install timm")
        
        # XceptionNet feature extractor
        self.xception = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,  # XceptionNet output features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) where:
               B = batch size
               T = temporal dimension (number of frames)
               C = channels (3 for RGB)
               H, W = height and width of frames
               
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Reshape for batch processing through XceptionNet
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Extract features using XceptionNet
        features = self.xception(x)  # (B*T, 2048)
        
        # Reshape back to sequence format
        features = features.view(B, T, -1)  # (B, T, 2048)
        
        # Process through LSTM
        lstm_out, (hn, cn) = self.lstm(features)  # (B, T, hidden_size)
        
        # Use the last time step output
        output = lstm_out[:, -1, :]  # (B, hidden_size)
        
        # Final classification
        logits = self.fc(output)  # (B, num_classes)
        
        return logits


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
        return VideoSwinTransformer(num_classes=num_classes, **kwargs)
    elif model_type == "xception":
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


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model architecture.
    
    Args:
        model: PyTorch model instance
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_name': model.__class__.__name__,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
