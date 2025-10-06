# Diabetes-Prediction-Model
This project implements a machine learning model to predict diabetes based on patient health data using Support Vector Classification (SVC)

Project Overview
The model analyzes various health parameters to classify whether a person is diabetic or not. It uses a dataset containing 768 samples with 8 different health features.

Dataset Information
The dataset contains the following features:

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index

DiabetesPedigreeFunction: Diabetes pedigree function

Age: Age in years

Outcome: Target variable (0 = non-diabetic, 1 = diabetic)

Dataset Statistics:

Total samples: 768

Non-diabetic cases: 500

Diabetic cases: 268

Model Implementation
Libraries Used
numpy - Numerical computations

pandas - Data manipulation

scikit-learn - Machine learning algorithms and utilities

Preprocessing Steps
Data loading and exploration

Feature-target separation

Data standardization using StandardScaler

Train-test split (80-20 ratio) with stratification

Model Training
Algorithm: Support Vector Classifier (SVC) with linear kernel

Training Accuracy: 78.34%

Testing Accuracy: 77.27%

Usage
The model can be used to predict diabetes risk based on patient health parameters:

python
# Example prediction
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
prediction = model.predict(input_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
File Structure
diabetes_model.ipynb - Main Jupyter notebook containing the complete implementation

Dataset: /content/diabetes.csv (loaded from external source)

Key Findings
The dataset shows clear differences in feature means between diabetic and non-diabetic groups

Glucose levels, BMI, and age show significant differences between the two groups

The linear SVC model provides reasonable accuracy for diabetes prediction

Future Improvements
Experiment with different machine learning algorithms

Perform hyperparameter tuning

Address potential class imbalance

Add feature engineering and selection

Implement cross-validation for more robust evaluation

Requirements
Python 3.x

numpy

pandas

scikit-learn

This model serves as a foundation for diabetes risk assessment and can be further enhanced with additional data and advanced techniques.

