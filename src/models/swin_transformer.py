"""
Video Swin Transformer model for deepfake detection.
"""

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from .base import BaseDeepfakeModel


class VideoSwinTransformer(BaseDeepfakeModel):
    """
    Video Swin Transformer model for deepfake detection.
    
    This model uses a pre-trained Video Swin Transformer as the backbone and
    replaces the final classification head for binary deepfake detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        pretrained (bool): Whether to use pre-trained weights (default: True)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes)
        
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
