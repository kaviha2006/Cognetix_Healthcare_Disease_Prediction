import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import split_features_target, perform_split

def train_and_tune_models(data_path='data/diabetes.csv', model_save_path='model.pkl'):
    # 1. Load & Clean
    print("Loading and cleaning data...")
    df = load_data(data_path)
    df = clean_data(df)
    
    # 2. Split X, y
    X, y = split_features_target(df, 'Outcome')
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = perform_split(X, y)
    
    # 4. Define Models & Pipelines
    # Note: XGBoost handles NaNs but we already cleaned them. Scaling helps some models.
    
    pipelines = {
        'LogisticRegression': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ]),
        'SVM': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('model', SVC(random_state=42, probability=True))
        ]),
        'XGBoost': Pipeline([
             ('smote', SMOTE(random_state=42)),
             ('scaler', StandardScaler()),
             ('model', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
        ])
    }
    
    # 5. Define Hyperparameters
    param_grids = {
        'LogisticRegression': {
            'model__C': [0.1, 1, 10],
            'model__solver': ['liblinear']
        },
        'RandomForest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20]
        },
        'SVM': {
            'model__C': [1, 10],
            'model__kernel': ['rbf'],
            'model__gamma': ['scale']
        },
        'XGBoost': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    }
    
    best_score = 0
    best_model_name = ""
    results = {}
    best_estimators = [] 

    print("Starting Training & Tuning (Phase 2 with XGBoost)...")
    
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        grid = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print(f"Best params for {name}: {grid.best_params_}")
        print(f"Best CV score for {name}: {grid.best_score_:.4f}")
        
        # Save best estimator for voting
        results[name] = grid.best_estimator_
        # For Voting, we need the actual estimator object, not the pipeline, usually? 
        # Actually VotingClassifier can take Pipelines as estimators.
        best_estimators.append((name, grid.best_estimator_))
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model_name = name
            
    print(f"\nBest Individual Model: {best_model_name} with Accuracy: {best_score:.4f}")
    
    # 6. Ensemble Voting Classifier
    print("Training Ensemble Voting Classifier...")
    voting_clf = VotingClassifier(estimators=best_estimators, voting='soft')
    voting_clf.fit(X_train, y_train)
    voting_score = voting_clf.score(X_test, y_test)
    print(f"Voting Classifier Test Accuracy: {voting_score:.4f}")
    
    # Determine the absolute best model including Voting
    # Note: using CV score for selection, but Voting score is on Test set. 
    # Let's trust Voting is generally robust. We will save the Voting Classifier as the primary model
    # OR compare it. For simplicity in Phase 2, let's verify if Voting beats the individual best on Test set.
    
    individual_best_model = results[best_model_name]
    individual_test_score = individual_best_model.score(X_test, y_test)
    print(f"Best Individual Model Test Accuracy: {individual_test_score:.4f}")
    
    final_model = voting_clf if voting_score >= individual_test_score else individual_best_model
    final_name = "VotingClassifier" if voting_score >= individual_test_score else best_model_name
    
    print(f"Selected Final Model: {final_name}")

    # Save Best Model
    joblib.dump(final_model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return results, X_test, y_test

if __name__ == "__main__":
    import os
    if os.path.exists("healthcare_disease_prediction/data/diabetes.csv"):
        path = "healthcare_disease_prediction/data/diabetes.csv"
        save_path = "healthcare_disease_prediction/model.pkl"
    else:
        path = "data/diabetes.csv"
        save_path = "model.pkl"
        
    train_and_tune_models(path, save_path)
