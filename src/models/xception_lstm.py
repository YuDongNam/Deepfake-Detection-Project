"""
XceptionNet + LSTM model for deepfake detection.
"""

import torch
import torch.nn as nn
from .base import BaseDeepfakeModel


class XceptionLSTM(BaseDeepfakeModel):
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
        super().__init__(num_classes)
        
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
