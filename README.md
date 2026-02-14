# Deep Learning-based Prediction of Lifespan Degradation in Concrete Bridges Due to Iron Oxidation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-Computer%20Vision-green.svg)

## 📖 Abstract

This project implements a deep learning-based system for predicting the lifespan degradation of concrete bridges caused by iron oxidation. The system uses a multimodal approach combining computer vision (YOLOv8) for damage detection with a Keras-based multimodal model for predicting bridge safety and remaining useful life.

## 📑 Publication

This research has been published. Please cite:

**Deep Learning-based Prediction of Lifespan Degradation in Concrete Bridges Due to Iron Oxidation**  
*American Journal of Traffic and Transportation Engineering*  
DOI: https://www.sciencepublishinggroup.com/article/10.11648/j.ajtte.20251005.11

## 🏗️ Project Overview

The system consists of two main components:

### 1. Damage Detection (YOLOv8)
- Uses YOLOv8 object detection model to identify damage in bridge images
- Detects various types of structural damage including cracks, spalling, and deterioration
- Provides bounding boxes with confidence scores

### 2. Safety Prediction (Multimodal Model)
- Combines visual features from YOLO with tabular data
- Predicts bridge safety status (Safe/Unsafe)
- Estimates remaining useful life in years
- Provides degradation scores

### Input Features
- **Image Data**: Bridge images for visual inspection
- **Tabular Data**:
  - Crack width (mm)
  - Crack length (cm)
  - Bridge age (years)
  - Electrical resistivity
  - Half-cell potential
  - Delamination depth (mm)
  - Cover thickness (mm)
  - And more...

## 📁 Project Structure

```
FINAL PROJECT/
├── best.pt                          # YOLOv8 trained model
├── multimodal_model_full_24_unfrozen.h5  # Keras multimodal model
├── FINAL_CODE_V_12ipynb.ipynb       # Main prediction code
├── create_AI_base_train_image_csv.ipynb  # Data preparation
├── bridge_predictions_final.csv     # Sample predictions
├── report final.docx                # Project report
├── Video of working.mp4             # Demo video
└── README.md                        # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local environment

### Required Libraries
```
bash
pip install ultralytics tensorflow pillow opencv-python matplotlib ipywidgets
```

## 💻 Usage

### Running in Google Colab
1. Open `FINAL_CODE_V_12ipynb.ipynb` in Google Colab
2. Mount Google Drive (if models are stored there)
3. Run cells sequentially
4. Upload bridge images through the interface
5. Get predictions for safety and remaining life

### Running Locally
1. Clone the repository
2. Install dependencies
3. Update model paths in the notebook
4. Run the prediction interface

## 📊 Model Outputs

The system provides:
- **Classification**: Safe or Unsafe
- **Degradation Score**: 0-100 scale
- **Remaining Life**: Estimated years of service remaining

## 🔬 Research Details

### Methodology
1. **Data Collection**: Bridge inspection images and sensor data
2. **Object Detection**: YOLOv8 trained on bridge damage dataset
3. **Feature Extraction**: Combined visual and tabular features
4. **Prediction**: Multimodal neural network for degradation prediction

### Results
- Accurate damage detection using state-of-the-art YOLOv8
- Multimodal fusion improves prediction accuracy
- Provides actionable insights for bridge maintenance

## 📝 License

This project is for research and educational purposes.

## 📚 Citation

If you use this code in your research, please cite:

```
bibtex
@article{bridge_degradation_2025,
  title={Deep Learning-based Prediction of Lifespan Degradation in Concrete Bridges Due to Iron Oxidation},
  author={},
  journal={American Journal of Traffic and Transportation Engineering},
  volume={10},
  number={5},
  year={2025},
  publisher={Science Publishing Group}
}
```

## 👤 Author

HETKUMAR PATEL

*Last updated: 15 June 2025*
