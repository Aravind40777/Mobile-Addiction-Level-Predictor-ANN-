# Mobile Addiction Level Predictor using Artificial Neural Network (ANN)

## Overview

The **Mobile Addiction Level Predictor** is a Deep Learning regression project developed using an Artificial Neural Network (ANN). The model predicts a user's mobile addiction level based on various smartphone usage and behavioral features.

The project covers the complete machine learning pipeline, including data preprocessing, feature encoding, feature scaling, ANN model development, hyperparameter tuning using Optuna, and model evaluation.

---

## Project Links

- **GitHub Repository:** https://github.com/Aravind40777/Mobile-Addiction-Level-Predictor-ANN-
- **Live Demo:** https://mobileaddictionlevelpredictorapp.streamlit.app/

Visit the live application to interact with the Mobile Addiction Level Predictor built using Artificial Neural Networks (ANN). The app accepts user inputs, processes them through the trained deep learning model, and predicts the estimated mobile addiction level in real time.

## Project Objectives

- Analyze mobile usage data.
- Preprocess and clean the dataset.
- Encode categorical features.
- Scale numerical features.
- Build an Artificial Neural Network using TensorFlow and Keras.
- Optimize the ANN using Optuna.
- Evaluate model performance using regression metrics.
- Save the trained model and scaler for future predictions.

---

## Dataset

The project uses the **Phone Addiction Dataset**, which contains information about users' smartphone usage patterns and behavioral characteristics.

The dataset includes demographic, lifestyle, and mobile usage features used to predict the **Mobile Addiction Level**.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- Optuna
- Joblib

---

## Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
optuna
joblib
```

---

## Project Workflow

1. Import Required Libraries
2. Load Dataset
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Label Encoding
6. Feature Scaling using StandardScaler
7. Train-Test Split
8. Build ANN Model
9. Train the Model
10. Hyperparameter Tuning using Optuna
11. Evaluate the Model
12. Save Model and Scaler

---

## ANN Architecture

The model consists of:

- Input Layer
  - 64 Neurons
  - ReLU Activation

- Hidden Layer 1
  - 32 Neurons
  - ReLU Activation

- Hidden Layer 2
  - 16 Neurons
  - ReLU Activation

- Output Layer
  - 1 Neuron (Regression Output)

Additional Techniques:

- Dropout Layer
- Early Stopping
- Adam Optimizer

---

## Hyperparameter Tuning

The project uses **Optuna** to optimize:

- Number of neurons
- Dropout rate
- Learning rate

The objective is to minimize the validation loss and improve model performance.

---

## Model Evaluation

The ANN model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help measure the prediction accuracy of the regression model.

---

## Files Included

```
Mobile-Addiction-Level-Predictor-ANN/
│
├── modile_addiction_predictor.ipynb
├── scaler.pkl
├── model.keras
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Aravind40777/Mobile-Addiction-Level-Predictor-ANN-.git
```

Move into the project folder:

```bash
cd Mobile-Addiction-Level-Predictor-ANN-
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## Future Enhancements

- Develop a Streamlit web application
- Deploy the model on Streamlit Cloud
- Improve feature engineering
- Experiment with deeper neural networks
- Compare ANN performance with other machine learning algorithms

---

## Applications

- Digital Well-being Analysis
- Smartphone Usage Monitoring
- Behavioral Analytics
- Academic Research
- Mental Health Studies
- User Habit Prediction

---

## Author

**Aravind Bhukya**

GitHub:  
https://github.com/Aravind40777

LinkedIn:  
https://www.linkedin.com/in/aravindbhukya06/

---

## License

This project is intended for educational and learning purposes.
