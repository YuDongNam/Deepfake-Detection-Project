"""
Main entry point for the deepfake detection project.

This script provides a command-line interface for training and evaluating
deepfake detection models using either Video Swin Transformer or XceptionNet+LSTM.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config import config
from src.data_loader import create_data_loaders
from src.models import create_model, load_model, get_model_info
from src.train import train_with_config
from src.evaluate import evaluate_model_comprehensive, print_evaluation_summary


def setup_logging():
    """Setup logging configuration."""
    import logging
    
    # Create logs directory
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_CONFIG["log_level"]),
        format=config.LOGGING_CONFIG["log_format"],
        handlers=[
            logging.FileHandler(config.LOGS_DIR / config.LOGGING_CONFIG["log_file"]),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def train_model_command(args):
    """Train a deepfake detection model."""
    logger = setup_logging()
    logger.info(f"Starting training for {args.model} model")
    
    # Create directories
    config.create_directories()
    
    # Set device
    device = torch.device(config.get_device())
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=args.data_dir,
        model_type=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Get training configuration
    training_config = config.get_training_config(args.model)
    training_config.update({
        "device": device,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "save_path": args.save_path or training_config["save_path"]
    })
    
    # Train model
    logger.info("Starting model training...")
    best_model, history = train_with_config(
        model_type=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Save training history
    history_path = config.RESULTS_DIR / f"{args.model}_training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {}
        for key, values in history.history.items():
            json_history[key] = [float(v) for v in values]
        json.dump(json_history, f, indent=2)
    
    logger.info(f"Training completed. Model saved to: {training_config['save_path']}")
    logger.info(f"Training history saved to: {history_path}")
    
    # Print model info
    model_info = get_model_info(best_model)
    logger.info(f"Model info: {model_info}")
    
    return best_model, history


def evaluate_model_command(args):
    """Evaluate a trained deepfake detection model."""
    logger = setup_logging()
    logger.info(f"Starting evaluation for {args.model} model")
    
    # Set device
    device = torch.device(config.get_device())
    logger.info(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    logger.info(f"Loading model from: {args.model_path}")
    model = load_model(
        model_path=args.model_path,
        model_type=args.model,
        device=device
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=args.data_dir,
        model_type=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate model
    logger.info("Starting model evaluation...")
    results = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.threshold
    )
    
    # Print evaluation summary
    print_evaluation_summary(results)
    
    # Save results
    results_path = config.RESULTS_DIR / f"{args.model}_evaluation_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        "sequence_metrics": {k: v for k, v in results["sequence_metrics"].items() 
                           if k not in ["predictions", "targets", "probabilities"]},
        "video_metrics": results["video_metrics"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "num_misclassified": len(results["misclassified_samples"])
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {results_path}")
    
    return results


def preprocess_data_command(args):
    """Preprocess raw video data to extract faces."""
    logger = setup_logging()
    logger.info("Starting data preprocessing...")
    
    from src.data_loader import extract_faces_from_videos
    
    # Create output directories
    config.PATHS["cropped_original"].mkdir(parents=True, exist_ok=True)
    config.PATHS["cropped_manipulated"].mkdir(parents=True, exist_ok=True)
    
    # Extract faces from original videos
    if args.original_dir and os.path.exists(args.original_dir):
        logger.info("Extracting faces from original videos...")
        extract_faces_from_videos(
            video_dir=args.original_dir,
            save_dir=str(config.PATHS["cropped_original"]),
            num_frames=config.DATASET_CONFIG["face_extraction_frames"]
        )
    
    # Extract faces from manipulated videos
    if args.manipulated_dir and os.path.exists(args.manipulated_dir):
        logger.info("Extracting faces from manipulated videos...")
        extract_faces_from_videos(
            video_dir=args.manipulated_dir,
            save_dir=str(config.PATHS["cropped_manipulated"]),
            num_frames=config.DATASET_CONFIG["face_extraction_frames"]
        )
    
    logger.info("Data preprocessing completed!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection using Vision Transformers and CNN-LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Video Swin Transformer model
  python main.py train --model swin --data_dir /path/to/data --epochs 30

  # Train XceptionNet+LSTM model
  python main.py train --model xception --data_dir /path/to/data --epochs 40

  # Evaluate trained model
  python main.py evaluate --model swin --model_path saved_models/best_swin_model.pth --data_dir /path/to/data

  # Preprocess raw video data
  python main.py preprocess --original_dir /path/to/original/videos --manipulated_dir /path/to/manipulated/videos
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a deepfake detection model')
    train_parser.add_argument('--model', choices=['swin', 'xception'], required=True,
                             help='Model architecture to train')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Path to the dataset directory')
    train_parser.add_argument('--epochs', type=int, default=30,
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=None,
                             help='Batch size (default: from config)')
    train_parser.add_argument('--learning_rate', type=float, default=None,
                             help='Learning rate (default: from config)')
    train_parser.add_argument('--patience', type=int, default=None,
                             help='Early stopping patience (default: from config)')
    train_parser.add_argument('--save_path', type=str, default=None,
                             help='Path to save the trained model')
    train_parser.add_argument('--num_workers', type=int, default=2,
                             help='Number of data loading workers')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', choices=['swin', 'xception'], required=True,
                            help='Model architecture')
    eval_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to the trained model file')
    eval_parser.add_argument('--data_dir', type=str, required=True,
                            help='Path to the dataset directory')
    eval_parser.add_argument('--batch_size', type=int, default=None,
                            help='Batch size (default: from config)')
    eval_parser.add_argument('--threshold', type=float, default=0.4,
                            help='Classification threshold')
    eval_parser.add_argument('--num_workers', type=int, default=2,
                            help='Number of data loading workers')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw video data')
    preprocess_parser.add_argument('--original_dir', type=str,
                                  help='Path to original videos directory')
    preprocess_parser.add_argument('--manipulated_dir', type=str,
                                  help='Path to manipulated videos directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set default values from config
    if hasattr(args, 'batch_size') and args.batch_size is None:
        data_loader_config = config.get_data_loader_config(args.model)
        args.batch_size = data_loader_config['batch_size']
    
    if hasattr(args, 'learning_rate') and args.learning_rate is None:
        training_config = config.get_training_config(args.model)
        args.learning_rate = training_config['learning_rate']
    
    if hasattr(args, 'patience') and args.patience is None:
        training_config = config.get_training_config(args.model)
        args.patience = training_config['patience']
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Execute command
        if args.command == 'train':
            train_model_command(args)
        elif args.command == 'evaluate':
            evaluate_model_command(args)
        elif args.command == 'preprocess':
            preprocess_data_command(args)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
