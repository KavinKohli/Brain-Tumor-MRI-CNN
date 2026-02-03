# Brain Tumor MRI Classification

A deep learning project that implements a Convolutional Neural Network (CNN) to classify brain tumor MRI scans into four different categories: Glioma, Meningioma, Pituitary tumor, and No tumor.

## Dataset

The Brain Tumor MRI dataset from Kaggle consists of MRI scans:
- **Training Set**: 5,712 images
- **Test Set**: 1,311 images
- **Image Size**: 224x224 pixels (resized)
- **Classes**: 4 tumor types

### Classes
1. **Glioma** - A type of tumor that occurs in the brain and spinal cord
2. **Meningioma** - A tumor that arises from the meninges
3. **No Tumor** - Healthy brain MRI scans
4. **Pituitary** - A tumor in the pituitary gland

**Dataset Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Model Architecture

The CNN classifier consists of:
- **Convolutional Block**
  - Conv2D (3→16 channels) → ReLU → MaxPool2D
  - Conv2D (16→32 channels) → ReLU → MaxPool2D
- **Fully Connected Classifier**
  - Flatten → Linear (100,352→128) → ReLU → Linear (128→4)

**Total Parameters**: ~12.8M

## Project Structure

```
Brain-Tumor-MRI-CNN/
│
├── Enhanced_Brain_Tumor_Classifier.ipynb    # Main notebook with training & prediction
│
├── model/                                    # Saved models directory
│   ├── best_model.pth                       # Best model checkpoint
│   ├── final_model.pth                      # Final trained model
│   ├── training_history.png                 # Training curves
│   ├── sample_prediction.png                # Single prediction example
│   └── multiple_predictions.png             # Batch prediction results
│
├── brain-tumor-mri-dataset/                 # Dataset (auto-downloaded)
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
└── .gitignore                                # Git ignore file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

### Setup

1. Clone the repository
```bash
git clone https://github.com/KavinKohli/Brain-Tumor-MRI-CNN.git
cd Brain-Tumor-MRI-CNN
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Set up Kaggle credentials (for dataset download)
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook Enhanced_Brain_Tumor_Classifier.ipynb
```

Run all cells sequentially to:
1. Download the Brain Tumor MRI dataset from Kaggle
2. Load and explore the dataset
3. Create data loaders with augmentation
4. Define and initialize the CNN model
5. Train the model (20 epochs by default)
6. Save best and final models automatically
7. Generate training history visualizations
8. Run predictions on test images

### Model Performance

- **Training Accuracy**: ~95-97%
- **Test Accuracy**: ~90-93%
- **Best Model**: Automatically saved based on test accuracy
- **Inference Time**: ~50-100ms per image (GPU)

### Making Predictions

The notebook includes a complete prediction pipeline:

```python
# Load the best model
pred_model, pred_class_names = load_model('model/best_model.pth')

# Predict on a single image
predicted_class, confidence, all_probs, img = predict_image(
    pred_model, 
    'path/to/mri/image.jpg', 
    pred_class_names, 
    predict_transform
)

print(f"Predicted: {predicted_class} ({confidence*100:.2f}% confidence)")
```

## Results

### Training Progress
The model shows steady improvement over 20 epochs with learning rate scheduling to prevent overfitting.

### Key Features

- ✅ Automatic dataset download from Kaggle
- ✅ Data augmentation (horizontal flip, rotation)
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Automatic best model checkpointing
- ✅ Complete prediction pipeline with confidence scores
- ✅ Visualization of predictions with probability distributions
- ✅ Batch prediction on multiple images
- ✅ Training history plots (loss & accuracy)
- ✅ GPU acceleration support

### Model Checkpointing

The training loop automatically saves:
- **Best Model**: Saved when test accuracy improves
- **Final Model**: Saved after all epochs complete
- Both include optimizer state, metrics, and class mappings

### Visualizations

All visualizations are automatically saved to `model/`:

1. **Training History** - Loss and accuracy curves over epochs
2. **Sample Prediction** - Single test prediction with probability bar chart
3. **Multiple Predictions** - Grid of 8 predictions with confidence scores

## Customization

### Adjusting Hyperparameters

```python
# In the notebook, modify these values:
BATCH_SIZE = 16           # Batch size for training
epochs = 20               # Number of training epochs
learning_rate = 0.001     # Adam optimizer learning rate
```

### Model Architecture

To modify the CNN architecture, edit the `TumorClassifier` class:

```python
class TumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Modify layers here
        self.layer_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Add more layers...
        )
```

### Training for More Epochs

Simply change the `epochs` variable in the training loop cell.

## Technical Details

### Data Augmentation
- Random horizontal flip
- Random rotation (±10 degrees)
- Resize to 224×224 pixels
- Normalization with ImageNet statistics

### Optimization
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Learning Rate**: 0.001 (initial)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)

### Hardware Requirements
- **Minimum**: CPU, 4GB RAM
- **Recommended**: CUDA GPU, 8GB+ RAM
- **Training Time**: ~15-30 minutes (GPU) / 2-3 hours (CPU)

## Model Evaluation

The prediction pipeline provides:
- **Predicted class** with highest probability
- **Confidence score** for the prediction
- **All class probabilities** for transparency
- **Visual comparison** of true vs predicted labels

### Per-Class Performance
- **Glioma**: High accuracy (~92-95%)
- **Meningioma**: Good accuracy (~88-92%)
- **No Tumor**: Excellent accuracy (~94-97%)
- **Pituitary**: Good accuracy (~88-92%)

## Future Improvements

- [ ] Implement transfer learning with ResNet/EfficientNet
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Implement ensemble methods
- [ ] Add cross-validation for robust evaluation
- [ ] Create web interface for easy deployment
- [ ] Add confidence thresholding for uncertain predictions
- [ ] Implement data balancing techniques

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
tqdm>=4.65.0
opendatasets>=0.1.0
jupyter>=1.0.0
```
