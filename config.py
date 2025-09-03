"""
Configuration file for deepfake detection project.

This module contains all hyperparameters, file paths, and model configurations
for the deepfake detection project.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for deepfake detection project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Dataset configuration
    DATASET_CONFIG = {
        "frame_num": 72,  # Number of consecutive frames to extract
        "min_video_length": 4,  # Minimum video length in seconds
        "face_extraction_frames": 8,  # Number of frames to extract for face detection
        "train_val_split": 0.2,  # Validation split ratio
        "random_state": 42,  # Random state for reproducibility
    }
    
    # Data loading configuration
    DATA_LOADER_CONFIG = {
        "batch_size": 4,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle_train": True,
        "shuffle_val": False,
        "shuffle_test": False,
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        "swin": {
            "model_type": "swin",
            "input_size": 224,
            "num_classes": 2,
            "pretrained": True,
            "model_kwargs": {}
        },
        "xception": {
            "model_type": "xception",
            "input_size": 299,
            "num_classes": 2,
            "pretrained": True,
            "model_kwargs": {
                "hidden_size": 256,
                "num_layers": 1
            }
        }
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        "num_epochs": 30,
        "learning_rate": 1e-4,
        "patience": 10,
        "threshold": 0.4,  # Binary classification threshold
        "use_focal_loss": True,
        "focal_alpha": 0.1,
        "focal_gamma": 2.0,
        "optimizer": "adam",
        "weight_decay": 1e-5,
        "scheduler": None,  # Can be "cosine", "step", or None
        "scheduler_params": {}
    }
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        "threshold": 0.4,
        "max_misclassified_samples": 20,
        "save_predictions": True,
        "save_confusion_matrix": True,
        "generate_plots": True
    }
    
    # Device configuration
    DEVICE_CONFIG = {
        "use_cuda": True,
        "cuda_device": 0,  # GPU device index
        "mixed_precision": False,  # Use automatic mixed precision
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        "log_level": "INFO",
        "log_file": "training.log",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "save_tensorboard": True,
        "tensorboard_dir": "runs"
    }
    
    # File paths
    PATHS = {
        "original_data": DATA_DIR / "original",
        "manipulated_data": DATA_DIR / "manipulated",
        "cropped_original": DATA_DIR / "cropped" / "original",
        "cropped_manipulated": DATA_DIR / "cropped" / "manipulated",
        "best_swin_model": SAVED_MODELS_DIR / "best_swin_model.pth",
        "best_xception_model": SAVED_MODELS_DIR / "best_xception_model.pth",
        "results_file": RESULTS_DIR / "evaluation_results.json",
        "config_backup": RESULTS_DIR / "config_backup.yaml"
    }
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model type.
        
        Args:
            model_type: Type of model ("swin" or "xception")
            
        Returns:
            Dictionary containing model configuration
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.MODEL_CONFIGS.keys())}")
        
        return cls.MODEL_CONFIGS[model_type].copy()
    
    @classmethod
    def get_training_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get training configuration for a specific model type.
        
        Args:
            model_type: Type of model ("swin" or "xception")
            
        Returns:
            Dictionary containing training configuration
        """
        config = cls.TRAINING_CONFIG.copy()
        model_config = cls.get_model_config(model_type)
        
        # Add model-specific save path
        if model_type == "swin":
            config["save_path"] = str(cls.PATHS["best_swin_model"])
        elif model_type == "xception":
            config["save_path"] = str(cls.PATHS["best_xception_model"])
        
        return config
    
    @classmethod
    def get_data_loader_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get data loader configuration for a specific model type.
        
        Args:
            model_type: Type of model ("swin" or "xception")
            
        Returns:
            Dictionary containing data loader configuration
        """
        config = cls.DATA_LOADER_CONFIG.copy()
        model_config = cls.get_model_config(model_type)
        
        # Adjust batch size based on model type
        if model_type == "xception":
            config["batch_size"] = 2  # XceptionNet+LSTM uses smaller batch size
        
        return config
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.SAVED_MODELS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
            cls.PATHS["cropped_original"],
            cls.PATHS["cropped_manipulated"]
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls) -> str:
        """
        Get the device string for PyTorch.
        
        Returns:
            Device string ("cuda" or "cpu")
        """
        import torch
        
        if cls.DEVICE_CONFIG["use_cuda"] and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if required directories exist or can be created
            cls.create_directories()
            
            # Validate model configurations
            for model_type in cls.MODEL_CONFIGS:
                cls.get_model_config(model_type)
                cls.get_training_config(model_type)
                cls.get_data_loader_config(model_type)
            
            # Validate paths
            for path_name, path in cls.PATHS.items():
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Default configuration instance
config = Config()

# Validate configuration on import
if not config.validate_config():
    print("Warning: Configuration validation failed. Please check your configuration.")


# Example usage and configuration overrides
def get_config_for_experiment(experiment_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary containing experiment-specific configuration
    """
    base_config = {
        "dataset": config.DATASET_CONFIG.copy(),
        "data_loader": config.DATA_LOADER_CONFIG.copy(),
        "training": config.TRAINING_CONFIG.copy(),
        "evaluation": config.EVALUATION_CONFIG.copy(),
        "device": config.DEVICE_CONFIG.copy(),
        "logging": config.LOGGING_CONFIG.copy(),
        "paths": {k: str(v) for k, v in config.PATHS.items()}
    }
    
    # Experiment-specific overrides
    if experiment_name == "swin_baseline":
        base_config["model"] = config.get_model_config("swin")
        base_config["training"]["num_epochs"] = 50
        base_config["training"]["learning_rate"] = 5e-5
        
    elif experiment_name == "xception_baseline":
        base_config["model"] = config.get_model_config("xception")
        base_config["training"]["num_epochs"] = 40
        base_config["training"]["learning_rate"] = 1e-4
        
    elif experiment_name == "swin_large_batch":
        base_config["model"] = config.get_model_config("swin")
        base_config["data_loader"]["batch_size"] = 8
        base_config["training"]["learning_rate"] = 2e-4
        
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return base_config
