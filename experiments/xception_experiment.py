"""
Experiment script for XceptionNet+LSTM model.
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


def run_xception_experiment():
    """Run XceptionNet+LSTM experiment with specific configurations."""
    
    # Xception-specific configuration
    xception_config = {
        "model_type": "xception",
        "num_classes": 2,
        "num_epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 2,  # Smaller batch size for XceptionNet+LSTM
        "patience": 12,
        "use_focal_loss": True,
        "focal_alpha": 0.1,
        "focal_gamma": 2.0,
        "model_kwargs": {
            "hidden_size": 256,
            "num_layers": 1
        },
        "save_path": "saved_models/best_xception_experiment.pth"
    }
    
    print("🚀 Starting XceptionNet+LSTM Experiment")
    print(f"Configuration: {xception_config}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir="data",
        model_type="xception",
        batch_size=xception_config["batch_size"]
    )
    
    # Train model
    best_model, history = train_with_config(
        model_type="xception",
        train_loader=train_loader,
        val_loader=val_loader,
        config=xception_config
    )
    
    # Evaluate model
    results = evaluate_model_comprehensive(
        model=best_model,
        test_loader=test_loader,
        device="cuda" if config.get_device() == "cuda" else "cpu"
    )
    
    # Print comprehensive performance summary
    print("\n" + "="*80)
    print("🎯 XCEPTIONNET+LSTM - FINAL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Video-level metrics (main results)
    video_metrics = results['video_metrics']
    print(f"📊 Video-level Performance (Multi-face Test Set):")
    print(f"   Accuracy:  {video_metrics['video_accuracy']:.4f}")
    print(f"   F1-Score:  {video_metrics['video_f1_score']:.4f}")
    print(f"   AUC:       {video_metrics['video_auc']:.4f}")
    print(f"   MCC:       {video_metrics['video_mcc']:.4f}")
    print(f"   Precision: {video_metrics['video_precision']:.4f}")
    print(f"   Recall:    {video_metrics['video_recall']:.4f}")
    print(f"   PR-AUC:    {video_metrics['video_pr_auc']:.4f}")
    
    print("\n✅ XceptionNet+LSTM Experiment Completed!")
    return results


if __name__ == "__main__":
    run_xception_experiment()
