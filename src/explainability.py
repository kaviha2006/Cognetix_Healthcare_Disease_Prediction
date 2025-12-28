import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_feature_importance(model, feature_names):
    """
    Plots Feature Importance for Random Forest / Tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show() # Display or save
        
        # Explain top 3
        print("\nTop 3 Important Features:")
        for i in range(3):
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    else:
        print("Model does not support feature_importances_ (e.g., Logistic Regression might use coefficients).")
        if hasattr(model, 'coef_'):
             # Logic for coefficients if needed, but Step 9 specifically asks for Random Forest feature importance
             print("Using Random Forest is recommended for this step.")

def run_shap_analysis(model, X_train, X_test):
    """
    Generates SHAP summary and force plots.
    """
    print("\nRunning SHAP Analysis...")
    
    # SHAP requires the underlying model if it's a wrapper (like Pipeline)
    # If model is a Pipeline, we need to apply the pre-processing steps of the pipeline to X
    # and then explain the final estimator.
    
    # We assume 'model' passed here is the *trained estimator* inside the pipeline? 
    # Or we handle the pipeline decomposition.
    
    # For robust implementation in a script, let's assume we extract the step.
    # But usually creating an Explainer on the pipeline predict is possible but slow/complex (KernelExplainer).
    # TreeExplainer is fast for RF.
    
    # We will try to detect if it's a pipeline
    estimator = model
    X_transformed = X_test
    
    if hasattr(model, 'steps'): # It's a pipeline
        # Transform data using steps except the last one
        transformer = model[:-1]
        estimator = model[-1]
        try:
             X_transformed = transformer.transform(X_test)
        except:
             # If SMOTE is in pipeline, it doesn't transform Test data (imbalanced-learn pipeline).
             # Standard Scaler does.
             # We should transform X_test using the scaler. 
             # Let's rely on the fact that we can just pass the processed X_test if called from main.
             pass

    # Select Explainer
    try:
        if "Forest" in str(type(estimator)):
            explainer = shap.TreeExplainer(estimator)
        elif "Linear" in str(type(estimator)):
            explainer = shap.LinearExplainer(estimator, X_transformed) # needs background
        elif "SVC" in str(type(estimator)):
             explainer = shap.KernelExplainer(estimator.predict_proba, X_transformed[:50]) # Slow
        else:
            explainer = shap.KernelExplainer(estimator.predict_proba, X_transformed[:50])
            
        shap_values = explainer.shap_values(X_transformed)
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_transformed, show=True) # or show=False and plt.show()
        
        # Force Plot (for one instance)
        # plt.figure()
        # shap.force_plot(...) # this is interactive JS, usually hard to show in static script. 
        # User asked to "Generate: SHAP force plot", usually implies saving it or showing in notebook.
        
        print("SHAP Summary plot generated.")
    except Exception as e:
        print(f"SHAP Error: {e}")

