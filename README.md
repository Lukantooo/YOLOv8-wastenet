<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/logonav.png" width="25%"/>
</p>

# TACO - YOLOv8 Implementation

TACO is a growing image dataset of waste in the wild. It contains images of litter taken under diverse environments: woods, roads and beaches. These images are manually labeled and segmented according to a hierarchical taxonomy to train and evaluate object detection algorithms. 

This implementation uses **YOLOv8** for modern, efficient trash detection with enhanced class mapping and data preprocessing.

<div align="center">
  <div class="column">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/1.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/2.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/3.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/4.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/5.png" width="17%" hspace="3">
  </div>
</div>
</br>

## 🆕 What's New in This Implementation

- **YOLOv8 Integration**: Modern YOLO architecture for faster and more accurate detection
- **Smart Class Mapping**: 60+ original TACO classes consolidated into 11 practical categories
- **Enhanced Data Processing**: Collision-free filename handling and improved dataset utilization
- **Comprehensive Training**: Advanced augmentation and regularization techniques
- **Multi-Format Export**: Models exported for web (ONNX), mobile (TFLite), and server (PyTorch) deployment
- **Performance Analysis**: Detailed overfitting detection and train/validation comparison

## 📋 Publications

For more details check our paper: https://arxiv.org/abs/2003.06975

If you use this dataset and API in a publication, please cite us using:
```bibtex
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
```

## 🚀 Getting Started

### Requirements

**Python Version**: Python 3.12.3 (exact version required)

Install the required packages using the provided requirements.txt:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `ultralytics==8.3.154` (YOLOv8)
- `torch==2.7.1` (PyTorch)
- `opencv-python==4.11.0.86`
- `scikit-learn==1.7.0`
- `matplotlib==3.10.3`

### Download Dataset

To download the dataset images:
```bash
python3 download.py
```

Alternatively, download from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3587843.svg)](https://doi.org/10.5281/zenodo.3587843)

**Dataset Structure Expected:**
```
./data/
├── annotations.json
├── batch_1/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── batch_2/
│   ├── 000000.jpg
│   └── ...
└── ...
```

## 🎯 YOLOv8 Training

### Choose Your Implementation

**Option 1: With Class Mapping (Recommended)**
```bash
python YOLOmerged.py
```

**Option 2: Original TACO Classes**
```bash
python YOLO.py
```

### Key Differences

| Feature | YOLO.py | YOLOmerged.py |
|---------|---------|---------------|
| **Classes** | 60+ original TACO classes | 11 mapped categories |
| **Training Data** | Uses all original classes | Consolidates similar classes |
| **Performance** | Harder to train (class imbalance) | Better performance (balanced) |
| **Use Case** | Research/detailed analysis | Practical deployment |
| **Processing** | Direct TACO→YOLO conversion | Smart class mapping + conversion |

### Which Should You Use?

**Use `YOLOmerged.py` if:**
- ✅ You want better detection performance
- ✅ You need practical trash detection
- ✅ You're deploying in real applications
- ✅ You want to avoid class imbalance issues

**Use `YOLO.py` if:**
- 🔬 You need all original TACO categories
- 🔬 You're doing research requiring detailed classes
- 🔬 You want to compare with original TACO results
- 🔬 You have specific requirements for individual classes

## 🏷️ Class Mapping

The original 60+ TACO classes are intelligently mapped to 11 practical categories:

| YOLO ID | Category | Original Classes Included |
|---------|----------|---------------------------|
| 0 | **Bottle** | Clear plastic bottle, Glass bottle, Other plastic bottle |
| 1 | **Bottle cap** | Plastic bottle cap, Metal bottle cap |
| 2 | **Can** | Drink can, Food can |
| 3 | **Cigarette** | Cigarette |
| 4 | **Cup** | Paper cup, Disposable plastic cup, Foam cup, etc. |
| 5 | **Lid** | Plastic lid, Metal lid |
| 6 | **Plastic bag + wrapper** | Garbage bag, Carrier bags, Film, Wrappers, etc. |
| 7 | **Pop tab** | Pop tab |
| 8 | **Straw** | Plastic straw, Paper straw |
| 9 | **Other** | All remaining classes |

This mapping:
- ✅ Reduces class imbalance
- ✅ Focuses on most common litter types
- ✅ Improves detection accuracy
- ✅ Maintains practical utility

## 📊 Training Configuration

### Default Settings
- **Model**: YOLOv8l (large) for best accuracy
- **Epochs**: 200 with early stopping
- **Image Size**: 800px
- **Batch Size**: 8 (adjust based on GPU memory)
- **Split**: 80% train, 20% validation

### Advanced Features
- **Data Augmentation**: Best results were achieved with using only the standard augmentation techniques provided by YOLOv8. (https://docs.ultralytics.com/de/modes/train/#train-settings)

## 🎯 Model Performance

Expected performance metrics:
- **mAP50**: 0.394 (IoU threshold 0.5)
- **mAP50-95**: 0.308 (IoU threshold 0.5:0.95)

Performance varies based on:
- Available training data
- Hardware capabilities
- Training duration
- Class distribution

## 📦 Model Export

Models are automatically exported in multiple formats:

```bash
./exported_models/
├── taco_model.pt          # PyTorch (server deployment)
├── taco_model.onnx        # ONNX (web deployment)
└── taco_model.tflite      # TensorFlow Lite (mobile)
```

### Deployment Options
- **🌐 Web Applications**: Use ONNX with ONNX.js
- **📱 Mobile Apps**: Use TFLite with TensorFlow Lite
- **🖥️ Server APIs**: Use PyTorch (.pt) format
- **☁️ Cloud Services**: Use ONNX or PyTorch

## 🔧 Customization

### Custom Test Images
```python
from ultralytics import YOLO

model = YOLO('path/to/your/model.pt')
results = model('path/to/test/image.jpg', conf=0.25)
```

### Adjust Confidence Threshold
```python
# Lower threshold = more detections (but more false positives)
results = model('image.jpg', conf=0.1)

# Higher threshold = fewer detections (but higher precision)
results = model('image.jpg', conf=0.5)
```

### Custom Class Mapping
Modify the `create_class_mapping()` function in the code to create your own category groups.

## 📁 File Structure

```
project/
├── requirements.txt           # Exact package versions
├── YOLO.py                   # Step-by-step implementation
├── YOLOmerged.py            # Complete pipeline
├── download.py              # Dataset download script
├── data/                    # TACO dataset
├── output/                  # YOLO format dataset
├── runs/detect/             # Training results
├── exported_models/         # Exported models
└── inference_results/       # Test results
```

## 🆘 Troubleshooting

### Common Issues

**GPU Memory Errors:**
```bash
# Reduce batch size
batch=4  # or batch=2
```

**Missing Images:**
- Ensure all batch folders are downloaded
- Check data directory structure
- Verify file permissions

**Low Performance:**
- Increase training epochs
- Use larger model (yolov8l.pt)
- Check data quality and annotations
- Ensure sufficient training data

**Class Imbalance:**
- Review class mapping
- Consider data augmentation
- Use weighted loss functions

## 📈 Performance Monitoring

The training script provides comprehensive monitoring:

- **Real-time Metrics**: mAP, precision, recall during training
- **Overfitting Detection**: Automatic train/validation comparison
- **Visual Results**: Confusion matrices, prediction samples
- **Export Validation**: Multi-format model testing


## 📊 Dataset Statistics

- **Total Images**: 1,500+ (varies by download)
- **Total Annotations**: 4,784+
- **Original Classes**: 60+ categories
- **Mapped Classes**: 11 practical categories
- **Environments**: Woods, roads, beaches, urban areas
- **Format**: COCO → YOLO conversion included

For more details and latest updates, visit: [tacodataset.org](http://tacodataset.org)

---

## 💡 Tips for Best Results

1. **Ensure Complete Dataset**: Verify all batch folders downloaded
2. **Use GPU Training**: Significantly faster than CPU
3. **Monitor Overfitting**: Check train vs validation metrics
4. **Experiment with Confidence**: Adjust thresholds for your use case
5. **Export Early**: Save models in multiple formats during training

**Ready to detect trash with YOLOv8! 🎯🗑️**