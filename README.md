
# Personal Protective Equipment Detection using YOLOv8

![Project Logo](https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/blob/master/static/images/1.jpg?raw=true)

## Table of Contents
1. Introduction
2. Prerequisites
3. Installation
4. Usage
5. Model Architecture (YOLOv8)
6. Dataset
7. Training
8. Evaluation
9. Results
10. Deployment with Flask
11. Future Improvements
12. Contributors
13. License

## 1. Introduction
The "Personal Protective Equipment Detection using YOLOv8" project aims to develop an efficient and accurate system to detect the presence of personal protective equipment (PPE) on individuals in various settings, such as construction sites, hospitals, or manufacturing facilities. The system utilizes the YOLOv8 object detection model, leveraging machine learning and computer vision techniques to automatically identify whether a person is wearing appropriate PPE, including items like helmets, masks, and safety vests.

## 2. Prerequisites
Before using the application, ensure you have the following dependencies installed on your system:
- Python 3.6+
- Flask
- OpenCV
- Pytorch
- NumPy
- Matplotlib

## 3. Installation
To set up the project on your local machine, follow these steps:
1. Clone the repository to your local machine using the following command:

```https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/```


2. Change into the project directory:
```
cd your-repo
```

3. Install the required Python packages using pip:
```pip install -r requirements.txt```


## 4. Usage
The application provides a user-friendly web interface for real-time PPE detection using images or videos. To use the application:

- Run the Flask app using the following command:
```python flaskapp.py ```

- Open your web browser and navigate to `http://localhost:5000`.
- Upload an image or provide a link to a video for PPE detection.
- Click the "Detect PPE" button to initiate the detection process.
- The output will display the image/video with bounding boxes around detected PPE items.

## 5. Model Architecture (YOLOv8)
The YOLOv8 model is a state-of-the-art object detection architecture that combines the best features from YOLOv3 and YOLOv4. It consists of a backbone network (e.g., Darknet-53) for feature extraction, a detection head with anchor boxes, and several YOLO layers for prediction. The model is implemented using PyTorch, providing fast and accurate detection.

## 6. Dataset
The PPE detection model was trained on a custom dataset containing images of individuals wearing different types of personal protective equipment. The dataset includes the following PPE categories:
- Helmet
- Mask
- Safety Vest

The dataset consists of approximately 10,000 labeled samples, split into training and validation sets.

## 7. Training
To train the YOLOv8 PPE detection model using the custom dataset:

- Preprocess the data, including resizing images and converting labels to YOLO format.
- Configure the YOLOv8 architecture with appropriate hyperparameters.
- Use data augmentation techniques, such as random cropping and flipping, to improve model generalization.
- Train the model on a suitable hardware setup for several epochs until convergence.

## 8. Evaluation
The model's performance was evaluated using several evaluation metrics, including:
- Precision
![Precision Curve](https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/blob/master/120_V8n/P_curve.png?raw=true)
- Recall
![Recall Curve](https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/blob/master/120_V8n/R_curve.png?raw=true)
- F1 Score
![F1 Confidence Curve](https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/blob/master/120_V8n/F1_curve.png?raw=true)


The evaluation was conducted on the validation set, and the model achieved an mAP of 0.85 for PPE detection.

## 9. Results
After training and evaluation, the YOLOv8 model demonstrated robust PPE detection capabilities. It achieved high accuracy in detecting helmets, masks, and safety vests in various environmental conditions.

Here is a visualization of the detection results on sample images:

![Sample Detection](https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/blob/master/120_V8n/val_batch0_pred.jpg?raw=true)


## 10. Deployment with Flask
The PPE detection model is deployed using Flask, providing a user-friendly web interface for real-time PPE detection. Flask enables seamless integration with the model, allowing users to upload images or provide links to videos and receive instant detection results through the web app.

## 11. Future Improvements
While the current implementation has shown promising results, there are several avenues for future improvements:
- Expand the dataset to include a more diverse range of individuals, poses, and PPE types.
- Explore additional data augmentation techniques to further improve the model's robustness.
- Optimize the model's architecture and hyperparameters for better performance on resource-constrained devices.





