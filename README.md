# 🏥 Healthcare Disease Prediction (Diabetes)

## 📌 Problem Statement
Diabetes is a chronic condition that affects millions worldwide. Early detection is crucial for management and prevention of complications. This project aims to build a Machine Learning system to predict the likelihood of diabetes in patients based on diagnostic measures such as Glucose level, BMI, Insulin, and Age.

## 📂 Dataset Description
**Source**: Pima Indians Diabetes Dataset (`data/diabetes.csv`)
**Target**: `Outcome` (0: Non-Diabetic, 1: Diabetic)
**Features**:
- Pregnancies
- Glucose (mg/dL)
- BloodPressure (mm Hg)
- SkinThickness (mm)
- Insulin (mu U/ml)
- BMI (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction (diabetes likelihood based on family history)
- Age (years)

## 🛠️ ML Pipeline Overview
1.  **Data Preprocessing**: 
    - Handling invalid zero values in Glucose, BP, BMI, Insulin by replacing with median.
    - Scale numerical features using `StandardScaler`.
2.  **Class Imbalance Handling**:
    - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target classes.
3.  **Model Training**:
    - Trained Logistic Regression, Random Forest, and SVM.
    - Used **GridSearchCV** for hyperparameter tuning.
4.  **Evaluation**:
    - Verified using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    - **Recall** was maximized to minimize false negatives (critical in healthcare).

## 🚀 Models Used
- **Logistic Regression**: Baseline linear model.
- **Random Forest Classifier**: Ensemble method, robust to overfitting.
- **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
- **XGBoost**: Gradient prediction for high accuracy.
- **Ensemble Voting Classifier**: Combines all models for maximum stability (Final Model).

## 📊 Evaluation Metrics
*Sample Metrics:*
- **Accuracy**: ~0.87 (Enhanced with Ensemble)
- **Precision**: ~0.85
- **Recall**: Optimized via threshold tuning.
- **F1-Score**: ~0.86
- **ROC-AUC**: ~0.92

## 🔍 Explainability (SHAP)
We used **SHAP (SHapley Additive exPlanations)** to interpret the model's predictions.

## 🌟 Data Science Refinements (Phase 2)
- **Ensemble Learning**: We use a Soft Voting Classifier to average probabilities from multiple robust models.
- **Sensitivity Analysis**: The App now features a **Sensitivity Threshold Slider**.
    - **Standard (0.5)**: Balanced Precision/Recall.
    - **High Sensitivity (< 0.5)**: Catch more potential cases (Crucial for screening).
    - **High Precision (> 0.5)**: Minimize false alarms.

## 💻 How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train Models**:
    ```bash
    python -m src.train_models
    ```
    This will generate `model.pkl`.
3.  **Run Streamlit App**:
    ```bash
    streamlit run app.py
    ```

## ⚠️ Limitations & Future Improvements
- **Data Size**: The dataset is small (768 entries). More data would improve generalization.
- **Demographics**: Restricted to Pima Indian heritage females; may not generalize to all populations.
- **Future**: Incorporate more diverse datasets and real-time API integration.

## Author
- Name: Kaviha R M
- Mail-id: kaviharavichandran2006@gmail.com
