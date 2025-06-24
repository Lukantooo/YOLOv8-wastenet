# 1 Introduction

Improper waste disposal presents a significant global and environmental threat, impacting ecosystems and public helath. Therefore it´s crucial to make waste management as effective as possible.
Artificial Intelligence, especially computer vision, can enable automated waste identification and classification systems. Such automated systems have the potential of revolutionizing waste management, leading to cleaner public spaces and more efficient resource utilization.
The goal of this project is to test the accuracy of a machine learning model that can classify images of garbage

# 2 Related Work

Deep-Learning-based waste detection in natural and urban environments
- https://www.sciencedirect.com/science/article/pii/S0956053X21006474
AquaVison: Automating the detection of waste in water bodies using deep transfer learning#
- https://www.sciencedirect.com/science/article/pii/S2666016420300244
Outdoor trash detection in natural environment using a deep learning model
- https://ieeexplore.ieee.org/abstract/document/10244010/

# 3 Methodology
## 3.1 General Methodology

This project utilizes a systematic approach to develop and evaluate image classification models for waste classification using the TACO dataset. The methodology includes data preparation, model selection, training and evaluation.

## 3.2 Data Understanding and Preparation

The dataset required conversion from COCO format to YOLO format. We also mapped the 60 old categories down to 10 more general classes:
- Bottle
- Bottle Cap
- Can
- Cigarette
- Cup
- Lid
- Plastic Bag + Wrapper
- Pop tab
- Straw
- Other

## 3.3 Modeling and Evaluation

After several experiments the optimal architecture was identified as follows:
- Architecture: YOLOv8 Large (yolov8l.pt)
- Training Duration: 200 epochs
- Image Size: 800×800 pixels
- Batch Size: 8
- Augmentation: None (due to dataset size limitations)
- Class Mapping: 10 consolidated categories
- Final Performance: mAP50: 0.394, mAP50-95: 0.308