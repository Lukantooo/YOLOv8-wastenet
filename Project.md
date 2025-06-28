# 1 Introduction

Improper waste disposal presents a significant global and environmental threat, impacting ecosystems and public helath. Therefore it´s crucial to make waste management as effective as possible.
Artificial Intelligence, especially computer vision, can enable automated waste identification and classification systems. Such automated systems have the potential of revolutionizing waste management, leading to cleaner public spaces and more efficient resource utilization.
The goal of this project is to test the accuracy of a machine learning model that can classify images of garbage

# 2 Related Work

Deep-Learning-based waste detection in natural and urban environments
- https://www.sciencedirect.com/science/article/pii/S0956053X21006474
Outdoor trash detection in natural environment using a deep learning model
- https://ieeexplore.ieee.org/abstract/document/10244010/
Our implementation is based on the following kaggle notebook:
- https://www.kaggle.com/code/arshnoor7389/taco-to-yolo-waste-detection-with-yolov8
TACO Dataset
- https://github.com/pedropro/TACO

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

# 4 Results
Artifacts Built:
- A complete object detection Pipeline for waste classification using the TACO dataset
- Custom conversion script from COCO to YOLO annotation format
- Multiple trained YOLO models (YOLOv8 Medium, Large; YOLOv12) 
- Visualizations including training curves, confusion matrix and sample prediction Outputs

Libraries and Tools Used:
- Ultralytics YOLOv8 (v8.3.153), PyTorch 2.7.1+cu126, Python 3.12.3
- NVIDIA CUDA on RTX 4070 SUPER GPU
- COCO and YOLO Annotation Tools, custom preprossesing scripts

Concept of the App
- A Computer vision-based solution for detecting and classifying waste objects in Images
- Supports use cases such as mobile litter detection apps, smart City Surveillance Integration and automated Recycling processes

Reuslts on unseen data
- Final model (YOLOv8 Large, 200 epochs) achieved:

mAP50: 0.394

mAP50-95: 0.308

Precision: 0.551

Recall: 0.383

- Best performance on categories like "Bottle" and "Can"; weakest on heterogeneous or small-object classes like "Other" and "Straw".
- Severe overfitting observed due to limited dataset size.


# 5 Discussion
The final model, developed using YOLOv8 on the TACO dataset, demonstrated strong performance on certain waste categories such as bottles and cans but struggled with heterogeneous or low-representation classes. The limited size of the dataset (833 images), restricted GPU resources, and overfitting highlight constraints in scalability and real-world deployment. From an ethical perspective, the system could reinforce biases in data (e.g., underrepresentation of certain waste types), raising concerns about transparency and fairness; its widespread use may also lead to unintended consequences in labor displacement or data misuse in smart city surveillance.


# 6 Conclusion
Future projects may implement real-time object detection (https://docs.ultralytics.com/de/models/yoloe/#introduction).
This was tricky to implement for the specific use case of trash detection and was out of scope for this project.