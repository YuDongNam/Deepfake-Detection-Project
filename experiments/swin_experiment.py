"""
Experiment script for Video Swin Transformer model.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import config
from src.data_loader import create_data_loaders
from src.models import create_model
from src.train import train_with_config
from src.evaluate import evaluate_model_comprehensive


def run_swin_experiment():
    """Run Video Swin Transformer experiment with specific configurations."""
    
    # Swin-specific configuration
    swin_config = {
        "model_type": "swin",
        "num_classes": 2,
        "num_epochs": 50,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "patience": 15,
        "use_focal_loss": True,
        "focal_alpha": 0.1,
        "focal_gamma": 2.0,
        "save_path": "saved_models/best_swin_experiment.pth"
    }
    
    print("🚀 Starting Video Swin Transformer Experiment")
    print(f"Configuration: {swin_config}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir="data",
        model_type="swin",
        batch_size=swin_config["batch_size"]
    )
    
    # Train model
    best_model, history = train_with_config(
        model_type="swin",
        train_loader=train_loader,
        val_loader=val_loader,
        config=swin_config
    )
    
    # Evaluate model
    results = evaluate_model_comprehensive(
        model=best_model,
        test_loader=test_loader,
        device="cuda" if config.get_device() == "cuda" else "cpu"
    )
    
    print("✅ Video Swin Transformer Experiment Completed!")
    return results


if __name__ == "__main__":
    run_swin_experiment()
