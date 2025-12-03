"""
Model training module with three distinct algorithms:
1. Logistic Regression (linear baseline)
2. Support Vector Machine (kernel-based non-linear)
3. Neural Network (deep learning)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import os
from datetime import datetime


def load_processed_data():
    """Load preprocessed train/test data."""
    print("Loading preprocessed data...")
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Test samples: {len(X_test):,}")
    print(f"✓ Features: {len(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test


def apply_transformations(X_train, X_test):
    """
    Apply data transformations for different algorithms.
    
    Transformations:
    1. Original: No transformation (baseline)
    2. Standardized: Z-score normalization (mean=0, std=1)
    3. MinMax: Scale to [0, 1] range
    
    Returns:
    --------
    dict
        Dictionary with different transformations
    """
    print("\n" + "="*80)
    print("APPLYING DATA TRANSFORMATIONS")
    print("="*80)
    
    transformations = {}
    
    # 1. Original (no transformation)
    print("\n1. Original (Untransformed)")
    print("   Purpose: Baseline performance")
    transformations['original'] = {
        'X_train': X_train.copy(),
        'X_test': X_test.copy(),
        'scaler': None
    }
    print("   ✓ Original data prepared")
    
    # 2. Standardization (Z-score normalization)
    print("\n2. Standardization (Z-score normalization)")
    print("   Purpose: Center features at mean=0, std=1")
    print("   Formula: z = (x - μ) / σ")
    print("   Application: Logistic Regression, SVM, Neural Network")
    
    scaler_standard = StandardScaler()
    X_train_standard = pd.DataFrame(
        scaler_standard.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_standard = pd.DataFrame(
        scaler_standard.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    transformations['standardized'] = {
        'X_train': X_train_standard,
        'X_test': X_test_standard,
        'scaler': scaler_standard
    }
    print("   ✓ Standardization complete")
    
    # 3. Min-Max Scaling
    print("\n3. Min-Max Scaling")
    print("   Purpose: Scale features to [0, 1] range")
    print("   Formula: x_scaled = (x - min) / (max - min)")
    print("   Application: Neural Network alternative")
    
    scaler_minmax = MinMaxScaler()
    X_train_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_minmax = pd.DataFrame(
        scaler_minmax.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    transformations['minmax'] = {
        'X_train': X_train_minmax,
        'X_test': X_test_minmax,
        'scaler': scaler_minmax
    }
    print("   ✓ Min-Max scaling complete")
    
    return transformations


def train_logistic_regression(X_train, y_train, transformation_name):
    """
    Train Logistic Regression model.
    
    Algorithm: Linear probabilistic classifier using logistic function
    Parameters: Default (C=1.0, max_iter=2000, solver='lbfgs')
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: LOGISTIC REGRESSION ({transformation_name})")
    print(f"{'='*80}")
    print("Algorithm: Linear probabilistic classifier")
    print("Parameters: C=1.0 (L2 regularization), max_iter=2000, solver='lbfgs'")
    
    start_time = datetime.now()
    
    model = LogisticRegression(
        random_state=42,
        max_iter=2000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✓ Training complete in {elapsed:.2f} seconds")
    
    return model


def train_svm(X_train, y_train, transformation_name):
    """
    Train Support Vector Machine with RBF kernel.
    
    Algorithm: Maximum margin classifier with kernel trick
    Kernel: RBF (Radial Basis Function) for non-linear decision boundary
    Parameters: Default (C=1.0, gamma='scale')
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: SUPPORT VECTOR MACHINE ({transformation_name})")
    print(f"{'='*80}")
    print("Algorithm: SVM with RBF kernel")
    print("Kernel: K(x_i, x_j) = exp(-γ ||x_i - x_j||²)")
    print("Parameters: C=1.0, gamma='scale', probability=True")
    
    start_time = datetime.now()
    
    model = SVC(
        kernel='rbf',
        random_state=42,
        probability=True,  # Enable probability estimates
        C=1.0,
        gamma='scale',
        cache_size=1000  # Increase cache size to speed up training
    )
    model.fit(X_train, y_train)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✓ Training complete in {elapsed:.2f} seconds")
    print(f"  • Support vectors: {model.n_support_.sum()}/{len(X_train)} ({model.n_support_.sum()/len(X_train)*100:.1f}%)")
    
    return model


def train_neural_network(X_train, y_train, transformation_name):
    """
    Train Multi-Layer Perceptron (Neural Network).
    
    Architecture:
    - Input layer: 11 features
    - Hidden layer 1: 64 neurons (ReLU activation)
    - Hidden layer 2: 32 neurons (ReLU activation)
    - Output layer: 1 neuron (Sigmoid activation)
    
    Parameters:
    - Optimizer: Adam (adaptive learning rate)
    - Early stopping: Yes (10 epochs patience)
    - Validation: 10% of training data
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: NEURAL NETWORK ({transformation_name})")
    print(f"{'='*80}")
    print("Architecture:")
    print("  Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)")
    print("Parameters: Adam optimizer, early_stopping=True, max_iter=500")
    print("Validation: 10% of training data")
    
    start_time = datetime.now()
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    model.fit(X_train, y_train)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✓ Training complete in {elapsed:.2f} seconds")
    print(f"  • Converged at epoch: {model.n_iter_}")
    print(f"  • Final training loss: {model.loss_:.4f}")
    
    return model


def save_model(model, name, transformation):
    """Save trained model to disk."""
    os.makedirs('models', exist_ok=True)
    filename = f'models/{name}_{transformation}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved: {filename}")


def main():
    """
    Main training pipeline.
    
    Trains 3 algorithms with different transformations:
    - Logistic Regression: original + standardized
    - SVM: original + standardized
    - Neural Network: standardized + minmax
    """
    print("="*80)
    print("CARDIOVASCULAR DISEASE CLASSIFICATION - MODEL TRAINING")
    print("="*80)
    print("\nThis script trains 3 machine learning algorithms:")
    print("  1. Logistic Regression (linear baseline)")
    print("  2. Support Vector Machine (non-linear kernel-based)")
    print("  3. Neural Network (deep learning)")
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Apply transformations
    transformations = apply_transformations(X_train, X_test)
    
    # Dictionary to store all trained models
    trained_models = {}
    
    print("\n" + "="*80)
    print("ALGORITHM 1: LOGISTIC REGRESSION")
    print("="*80)
    
    # Train Logistic Regression on original data
    X_tr = transformations['original']['X_train']
    model = train_logistic_regression(X_tr, y_train, 'original')
    trained_models['logistic_original'] = model
    save_model(model, 'logistic_regression', 'original')
    
    # Train Logistic Regression on standardized data
    X_tr = transformations['standardized']['X_train']
    model = train_logistic_regression(X_tr, y_train, 'standardized')
    trained_models['logistic_standardized'] = model
    save_model(model, 'logistic_regression', 'standardized')
    
    print("\n" + "="*80)
    print("ALGORITHM 2: SUPPORT VECTOR MACHINE")
    print("="*80)
    
    # Train SVM on original data
    X_tr = transformations['original']['X_train']
    model = train_svm(X_tr, y_train, 'original')
    trained_models['svm_original'] = model
    save_model(model, 'svm', 'original')
    
    # Train SVM on standardized data
    X_tr = transformations['standardized']['X_train']
    model = train_svm(X_tr, y_train, 'standardized')
    trained_models['svm_standardized'] = model
    save_model(model, 'svm', 'standardized')
    
    print("\n" + "="*80)
    print("ALGORITHM 3: NEURAL NETWORK")
    print("="*80)
    
    # Train Neural Network on standardized data
    X_tr = transformations['standardized']['X_train']
    model = train_neural_network(X_tr, y_train, 'standardized')
    trained_models['nn_standardized'] = model
    save_model(model, 'neural_network', 'standardized')
    
    # Train Neural Network on MinMax scaled data
    X_tr = transformations['minmax']['X_train']
    model = train_neural_network(X_tr, y_train, 'minmax')
    trained_models['nn_minmax'] = model
    save_model(model, 'neural_network', 'minmax')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTrained {len(trained_models)} models:")
    for key in trained_models.keys():
        print(f"  • {key}")
    print("\nAll models saved to: models/")


if __name__ == "__main__":
    main()