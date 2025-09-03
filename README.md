# Deepfake Video Detection using Vision Transformers and CNN-LSTM

A comprehensive deepfake detection system that employs two state-of-the-art architectures: **Video Swin Transformer** and **XceptionNet+LSTM** to combat the spread of manipulated media.

## 🎯 Project Overview

This project implements a robust deepfake detection system using the "Deep Fake Detection (DFD) Entire Original Dataset" from FaceForensics. The system is designed to detect manipulated videos through advanced computer vision and deep learning techniques.

### Key Features

- **Dual Architecture Support**: Video Swin Transformer and XceptionNet+LSTM models
- **Smart Data Splitting**: Single-face videos for training/validation, multi-face videos for testing
- **Advanced Frame Sampling**: 72 consecutive frames from videos longer than 4 seconds
- **Comprehensive Evaluation**: Both sequence-level and video-level performance metrics
- **Production Ready**: Clean, modular codebase with proper documentation

## 🏗️ Architecture

### 1. Video Swin Transformer
- **Backbone**: Pre-trained Video Swin Transformer (Swin3D-T)
- **Input**: 72 consecutive frames of 224×224 resolution
- **Features**: Leverages 3D attention mechanisms for temporal modeling
- **Normalization**: ImageNet standard normalization

### 2. XceptionNet+LSTM
- **Feature Extractor**: Pre-trained XceptionNet on ImageNet
- **Temporal Modeling**: LSTM with 256 hidden units
- **Input**: 72 consecutive frames of 299×299 resolution
- **Features**: CNN features + sequential processing

## 📊 Dataset

### Deep Fake Detection (DFD) Dataset
- **Original Videos**: 363 authentic videos
- **Manipulated Videos**: 3,068 deepfake videos
- **Source**: FaceForensics dataset
- **Preprocessing**: Face extraction using RetinaFace detection

### Data Splitting Strategy
- **Training & Validation**: Videos containing a single face
- **Test Set**: Videos containing multiple faces
- **Frame Sampling**: 72 consecutive frames from randomly selected starting points
- **Minimum Length**: Videos longer than 4 seconds only

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   ```bash
   # Note: You need to download the DFD dataset separately
   # Place it in the data/ directory with the following structure:
   # data/
   # ├── original/
   # │   └── (original video files)
   # └── manipulated/
   #     └── (manipulated video files)
   ```

## 📁 Project Structure

```
deepfake-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py                 # Configuration management
├── main.py                   # Main entry point
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── models.py             # Model architectures
│   ├── train.py              # Training logic
│   └── evaluate.py           # Evaluation metrics
├── notebooks/                # Jupyter notebooks
│   └── original_notebooks/   # Original development notebooks
├── saved_models/             # Trained model checkpoints
├── data/                     # Dataset (not included in repo)
├── results/                  # Training and evaluation results
└── logs/                     # Training logs
```

## 🎮 Usage

### Command Line Interface

The project provides a comprehensive CLI for all operations:

#### 1. Data Preprocessing
```bash
# Extract faces from raw videos
python main.py preprocess \
    --original_dir /path/to/original/videos \
    --manipulated_dir /path/to/manipulated/videos
```

#### 2. Training Models

**Video Swin Transformer:**
```bash
python main.py train \
    --model swin \
    --data_dir /path/to/processed/data \
    --epochs 30 \
    --batch_size 4 \
    --learning_rate 1e-4
```

**XceptionNet+LSTM:**
```bash
python main.py train \
    --model xception \
    --data_dir /path/to/processed/data \
    --epochs 40 \
    --batch_size 2 \
    --learning_rate 1e-4
```

#### 3. Model Evaluation
```bash
# Evaluate trained model
python main.py evaluate \
    --model swin \
    --model_path saved_models/best_swin_model.pth \
    --data_dir /path/to/processed/data \
    --threshold 0.4
```

### Programmatic Usage

```python
from src.data_loader import create_data_loaders
from src.models import create_model
from src.train import train_model
from src.evaluate import evaluate_model_comprehensive

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    root_dir="/path/to/data",
    model_type="swin"
)

# Create model
model = create_model("swin", num_classes=2)

# Train model
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30
)

# Evaluate model
results = evaluate_model_comprehensive(
    model=trained_model,
    test_loader=test_loader,
    device="cuda"
)
```

## ⚙️ Configuration

The project uses a centralized configuration system in `config.py`. Key parameters include:

### Model Configuration
```python
MODEL_CONFIGS = {
    "swin": {
        "input_size": 224,
        "num_classes": 2,
        "pretrained": True
    },
    "xception": {
        "input_size": 299,
        "hidden_size": 256,
        "num_layers": 1
    }
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "patience": 10,
    "use_focal_loss": True,
    "focal_alpha": 0.1,
    "focal_gamma": 2.0
}
```

## 📈 Results

### Performance Metrics

| Model | Accuracy | F1-Score | AUC | Precision | Recall |
|-------|----------|----------|-----|-----------|--------|
| Video Swin Transformer | 0.9374 | 0.9650 | 0.8359 | 0.9650 | 0.9650 |
| XceptionNet+LSTM | 0.9374 | 0.9648 | 0.8511 | 0.9688 | 0.9609 |

*Results on multi-face video test set (video-level evaluation)*

### Key Findings
- Both models achieve high accuracy (>93%) on the test set
- Video Swin Transformer shows better precision
- XceptionNet+LSTM demonstrates slightly higher AUC
- Models perform well on both single-face and multi-face scenarios

## 🔧 Advanced Features

### Loss Functions
- **Focal Loss**: Handles class imbalance effectively
- **Cross Entropy**: Standard classification loss
- **Customizable**: Easy to add new loss functions

### Data Augmentation
- **Spatial**: Random rotation, flipping, cropping
- **Temporal**: Frame sampling strategies
- **Color**: Brightness, contrast, saturation adjustments

### Evaluation Metrics
- **Sequence-level**: Individual frame sequence performance
- **Video-level**: Aggregated video performance
- **Comprehensive**: Accuracy, F1, AUC, Precision, Recall, MCC

## 🛠️ Development

### Adding New Models
1. Implement model class in `src/models.py`
2. Add configuration in `config.py`
3. Update data loader if needed
4. Test with training pipeline

### Custom Data Loaders
```python
class CustomDataset(Dataset):
    def __init__(self, ...):
        # Your implementation
    
    def __getitem__(self, idx):
        # Return (data, label, metadata)
```

### Extending Evaluation
```python
def custom_evaluation(model, test_loader, device):
    # Your custom evaluation logic
    return metrics
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{deepfake-detection-2024,
  title={Deepfake Video Detection using Vision Transformers and CNN-LSTM},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/deepfake-detection}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/ main.py config.py

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FaceForensics**: For providing the DFD dataset
- **PyTorch Team**: For the excellent deep learning framework
- **Swin Transformer**: For the innovative vision transformer architecture
- **XceptionNet**: For the efficient CNN architecture

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/deepfake-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/deepfake-detection/discussions)
- **Email**: your.email@example.com

## 🔮 Future Work

- [ ] Real-time inference pipeline
- [ ] Mobile deployment optimization
- [ ] Additional model architectures
- [ ] Cross-dataset evaluation
- [ ] Adversarial robustness testing
- [ ] Web interface for demo

---

**Note**: This project is for research and educational purposes. Please ensure you have proper permissions when working with video data and respect privacy and ethical guidelines.
