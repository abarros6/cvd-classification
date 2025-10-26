"""
Model evaluation module.
Calculates comprehensive metrics for all trained models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os


def load_model(model_name, transformation):
    """Load a trained model from disk."""
    filename = f'models/{model_name}_{transformation}.pkl'
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate comprehensive evaluation metrics.
    
    Metrics:
    - Accuracy: Overall correctness
    - Precision: Of predicted positives, how many are correct?
    - Recall: Of actual positives, how many did we find?
    - F1-Score: Harmonic mean of precision and recall
    - AUC-ROC: Area under ROC curve (threshold-independent)
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba)
    }


def print_confusion_matrix(y_true, y_pred, model_name):
    """Print confusion matrix in readable format."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{model_name}:")
    print(f"                Predicted")
    print(f"                No CVD    CVD")
    print(f"Actual  No CVD   {tn:5d}   {fp:5d}")
    print(f"        CVD      {fn:5d}   {tp:5d}")
    print(f"\nTrue Negatives:  {tn:5d}  |  False Positives: {fp:5d}")
    print(f"False Negatives: {fn:5d}  |  True Positives:  {tp:5d}")


def main():
    """
    Main evaluation pipeline.
    Evaluates all trained models and saves results.
    """
    print("="*80)
    print("CARDIOVASCULAR DISEASE CLASSIFICATION - MODEL EVALUATION")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    X_test_orig = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    print(f"✓ Test samples: {len(X_test_orig):,}")
    
    # Apply transformations to test data
    print("\nApplying transformations to test data...")
    X_train_orig = pd.read_csv('data/X_train.csv')
    
    # Standardization
    scaler_std = StandardScaler()
    scaler_std.fit(X_train_orig)
    X_test_std = pd.DataFrame(
        scaler_std.transform(X_test_orig),
        columns=X_test_orig.columns
    )
    
    # Min-Max scaling
    scaler_mm = MinMaxScaler()
    scaler_mm.fit(X_train_orig)
    X_test_mm = pd.DataFrame(
        scaler_mm.transform(X_test_orig),
        columns=X_test_orig.columns
    )
    print("✓ Transformations applied")
    
    # Store all results
    results = []
    
    # =========================================================================
    # ALGORITHM 1: LOGISTIC REGRESSION
    # =========================================================================
    print("\n" + "="*80)
    print("EVALUATING: LOGISTIC REGRESSION")
    print("="*80)
    
    # Logistic Regression - Original
    print("\n1. Logistic Regression (Original)")
    lr_orig = load_model('logistic_regression', 'original')
    y_pred = lr_orig.predict(X_test_orig)
    y_prob = lr_orig.predict_proba(X_test_orig)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob, 'Logistic Regression (Original)')
    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC-ROC:  {metrics['AUC-ROC']:.4f}")
    print_confusion_matrix(y_test, y_pred, 'Logistic Regression (Original)')
    
    # Logistic Regression - Standardized
    print("\n2. Logistic Regression (Standardized)")
    lr_std = load_model('logistic_regression', 'standardized')
    y_pred = lr_std.predict(X_test_std)
    y_prob = lr_std.predict_proba(X_test_std)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob, 'Logistic Regression (Standardized)')
    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC-ROC:  {metrics['AUC-ROC']:.4f}")
    print_confusion_matrix(y_test, y_pred, 'Logistic Regression (Standardized)')
    
    # =========================================================================
    # ALGORITHM 2: SUPPORT VECTOR MACHINE
    # =========================================================================
    print("\n" + "="*80)
    print("EVALUATING: SUPPORT VECTOR MACHINE")
    print("="*80)
    
    # SVM - Original
    print("\n1. SVM (Original)")
    svm_orig = load_model('svm', 'original')
    y_pred = svm_orig.predict(X_test_orig)
    y_prob = svm_orig.predict_proba(X_test_orig)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob, 'SVM (Original)')
    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC-ROC:  {metrics['AUC-ROC']:.4f}")
    print_confusion_matrix(y_test, y_pred, 'SVM (Original)')
    
    # SVM - Standardized
    print("\n2. SVM (Standardized)")
    svm_std = load_model('svm', 'standardized')
    y_pred = svm_std.predict(X_test_std)
    y_prob = svm_std.predict_proba(X_test_std)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob, 'SVM (Standardized)')
    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC-ROC:  {metrics['AUC-ROC']:.4f}")
    print_confusion_matrix(y_test, y_pred, 'SVM (Standardized)')
    
    # =========================================================================
    # ALGORITHM 3: NEURAL NETWORK
    # =========================================================================
    print("\n" + "="*80)
    print("EVALUATING: NEURAL NETWORK")
    print("="*80)
    
    # Neural Network - Standardized
    print("\n1. Neural Network (Standardized)")
    nn_std = load_model('neural_network', 'standardized')
    y_pred = nn_std.predict(X_test_std)
    y_prob = nn_std.predict_proba(X_test_std)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob, 'Neural Network (Standardized)')
    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC-ROC:  {metrics['AUC-ROC']:.4f}")
    print_confusion_matrix(y_test, y_pred, 'Neural Network (Standardized)')
    
    # Neural Network - MinMax
    print("\n2. Neural Network (MinMax)")
    nn_mm = load_model('neural_network', 'minmax')
    y_pred = nn_mm.predict(X_test_mm)
    y_prob = nn_mm.predict_proba(X_test_mm)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob, 'Neural Network (MinMax)')
    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC-ROC:  {metrics['AUC-ROC']:.4f}")
    print_confusion_matrix(y_test, y_pred, 'Neural Network (MinMax)')
    
    # =========================================================================
    # FINAL RESULTS SUMMARY
    # =========================================================================
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL MODELS")
    print("="*80)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_results.csv', index=False)
    print("\n✓ Results saved to: results/model_results.csv")
    
    # Identify best model
    best_idx = results_df['AUC-ROC'].idxmax()
    best_model = results_df.loc[best_idx]
    
    print("\n" + "="*80)
    print("BEST PERFORMING MODEL")
    print("="*80)
    print(f"\nModel: {best_model['Model']}")
    print(f"AUC-ROC: {best_model['AUC-ROC']:.4f}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall: {best_model['Recall']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nNext step: Run 'python src/visualization.py' to generate plots")


if __name__ == "__main__":
    main()