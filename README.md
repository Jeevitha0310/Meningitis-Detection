AI-Driven Early Detection and Classification of Meningitis Using Clinical and Imaging Data
Overview

This project presents an AI-based hybrid diagnostic system for the early detection and classification of meningitis using both clinical data and medical imaging data. The system combines machine learning models for structured clinical parameters and a Vision Transformer (ViT) model for MRI/CT image analysis. A fusion approach is used to integrate predictions from both data sources and improve diagnostic reliability.

The project also includes a Streamlit web application with user login, patient data entry, image upload, prediction display, and history storage using SQLite database.

Problem Statement

Meningitis is a serious infection affecting the protective membranes of the brain and spinal cord. Early and accurate diagnosis is critical for effective treatment and reducing complications. Traditional diagnosis may take time and can depend heavily on expert interpretation. This project aims to support faster and more reliable diagnosis by using artificial intelligence to analyze both clinical parameters and medical images.

Objectives
To develop a system for early detection of meningitis using AI.
To classify meningitis using clinical parameters and MRI/CT images.
To apply machine learning models on clinical data.
To apply a Vision Transformer (ViT) model on imaging data.
To combine both predictions using a fusion model.
To build a user-friendly web application for prediction and patient history storage.
Dataset Used
1. Clinical Dataset

The clinical dataset was collected from Kaggle and contains patient medical attributes such as:

Age
Gender
WBC Count
Protein Level
Glucose Level
Pathogen Presence
Hemoglobin
WBC Blood Count
Platelets
CRP Level
Diagnosis

This dataset is used to train the clinical machine learning models:

Random Forest
XGBoost
Logistic Regression
2. Imaging Dataset

The imaging dataset consists of MRI/CT brain images related to meningitis, collected manually from Bing image sources and organized into categories such as:

Bacterial
Viral

The images are preprocessed and used to train the Vision Transformer (ViT) model.

Project Architecture

The project follows a hybrid AI pipeline:

Clinical Data Input
Imaging Data Input
Data Preprocessing
Clinical data cleaning, encoding, normalization
Image resizing, RGB conversion, normalization
Clinical Model Training
Random Forest
XGBoost
Logistic Regression
Image Model Training
Vision Transformer (ViT)
Fusion Layer
Combines clinical and image predictions
Prediction Output
Storage in SQLite Database
Display in Streamlit Web Application
Algorithms / Models Used
Clinical Models
Random Forest
XGBoost
Logistic Regression
Imaging Model
Vision Transformer (ViT)
Final Prediction
Fusion / Hybrid Model combining clinical and imaging outputs
Features
Clinical data based meningitis prediction
MRI/CT image based meningitis classification
Hybrid fusion of clinical + imaging predictions
Streamlit web application interface
User login and authentication
Patient history storage using SQLite
Model evaluation with accuracy, precision, recall, and F1-score
Technologies Used
Programming Language
Python
Tools
Visual Studio Code
Streamlit
SQLite
GitHub
Libraries
pandas
numpy
scikit-learn
xgboost
torch
torchvision
timm
matplotlib
seaborn
pillow
sqlite3
pickle / joblib
Project Structure
MINI_PROJECT/
│── data/
│     └── meningitis.csv
│
│── images/
│     ├── bacterial_data/
│     ├── viral_data/
│     └── processed_images/
│
│── models/
│     ├── random_forest.pkl
│     ├── xgboost.pkl
│     ├── logistic.pkl
│     ├── label_encoder.pkl
│     ├── model_features.pkl
│     └── vit_model.pth
│
│── database/
│     └── meningitis.db
│
│── src/
│     ├── train_model.py
│     ├── train_image.py
│     ├── database.py
│     └── app.py
│
│── requirements.txt
│── README.md
Workflow
Clinical Data Workflow
Load clinical dataset
Remove unnecessary columns such as Patient_ID
Encode target labels
Apply one-hot encoding to categorical features
Split data into train/validation/test sets
Train Random Forest, XGBoost, and Logistic Regression models
Save trained models and encoders
Imaging Data Workflow
Collect MRI/CT images
Organize into bacterial and viral folders
Resize all images to 224 × 224
Convert images to RGB format
Train Vision Transformer model
Save trained image model
Fusion Workflow
Accept clinical inputs from user
Accept uploaded MRI/CT image
Predict clinical output using trained ML model
Predict image output using ViT model
Combine both outputs using fusion logic
Show final meningitis prediction
Store result in SQLite database
Evaluation Metrics

The project uses the following evaluation metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

These metrics are used to evaluate:

Clinical models
ViT image model
Fusion model
Streamlit Application

The Streamlit app provides:

User login and registration
Clinical data input form
MRI/CT image upload
Prediction results display
Patient history storage and retrieval
