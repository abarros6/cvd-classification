# Cardiovascular Disease Classification

A machine learning project that implements and compares three different algorithms for cardiovascular disease prediction using patient health data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (see Dataset section below)
# Place cardio_train.csv in data/ folder

# Run complete pipeline
python run_all.py
```

## Project Overview

This project compares three machine learning algorithms for binary classification of cardiovascular disease:

- **Logistic Regression** - Linear probabilistic baseline
- **Support Vector Machine** - RBF kernel for non-linear classification  
- **Neural Network** - Multi-layer perceptron with 2 hidden layers

Each algorithm is tested with different data transformations (original, standardized, min-max scaled) to analyze their impact on performance.

## Dataset

**Source:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle

**Requirements:**
1. Create Kaggle account if you don't have one
2. Download `cardio_train.csv` 
3. Place file in `data/cardio_train.csv`

**Dataset Details:**
- 70,000 patient records (original)
- **Reduced to 10,000 samples** for efficient training (8,000 train + 2,000 test)
- 11 features: age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity
- Binary target: CVD present (1) or absent (0)
- Balanced classes (50/50 split) maintained through stratified sampling

## Installation

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Complete Pipeline
Run all steps automatically:
```bash
python run_all.py
```

This executes:
1. Data preprocessing, cleaning, and dataset reduction
2. Model training (3 algorithms × multiple transformations)
3. Model evaluation and metrics calculation
4. Visualization generation

### Individual Steps
Run specific pipeline steps:
```bash
# Step 1: Preprocess data
python src/data_preprocessing.py

# Step 2: Train models (takes 1-2 minutes with reduced dataset)
python src/model_training.py

# Step 3: Evaluate models
python src/evaluation.py

# Step 4: Generate visualizations
python src/visualization.py
```

## Project Structure

```
cvd-classification/
├── data/                          # Dataset and processed data
│   ├── cardio_train.csv          # Original dataset (download required)
│   ├── X_train.csv               # Generated training features
│   ├── X_test.csv                # Generated test features
│   ├── y_train.csv               # Generated training labels
│   └── y_test.csv                # Generated test labels
│
├── src/                           # Source code modules
│   ├── data_preprocessing.py     # Data cleaning and splitting
│   ├── model_training.py         # Train all algorithms
│   ├── evaluation.py             # Calculate performance metrics
│   └── visualization.py          # Generate plots and figures
│
├── results/                       # Model performance results
│   └── model_results.csv         # Generated metrics table
│
├── figures/                       # Generated visualizations
│   ├── model_comparison.png      # Combined performance comparison
│   ├── confusion_matrices.png    # Combined confusion matrices
│   ├── roc_curves.png            # Combined ROC curves
│   ├── transformation_impact.png # Combined scaling effects
│   ├── results_table.png         # Summary table
│   └── individual_*.png          # Individual readable figures
│
├── models/                        # Trained model files
│   ├── logistic_regression_*.pkl # LR models
│   ├── svm_*.pkl                 # SVM models
│   └── neural_network_*.pkl      # NN models
│
├── requirements.txt               # Python dependencies
└── run_all.py                    # Master execution script
```

## Pipeline Details

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- Loads raw dataset (70,000 records)
- Converts age from days to years
- Calculates BMI from height/weight
- Removes physiologically implausible outliers (~1,600 records)
- **Reduces dataset to 10,000 samples** using stratified sampling for efficient training
- Splits data into 80% training / 20% testing (8,000 / 2,000 samples)
- Saves processed data for subsequent steps

### 2. Model Training (`src/model_training.py`)
Trains 6 model variants:
- **Logistic Regression:** Original + Standardized data
- **SVM (RBF kernel):** Original + Standardized data  
- **Neural Network:** Standardized + Min-Max scaled data

**Model Configurations:**
- Logistic Regression: Default scikit-learn parameters
- SVM: RBF kernel, C=1.0, gamma='scale'
- Neural Network: 2 hidden layers (64, 32 neurons), ReLU activation

### 3. Model Evaluation (`src/evaluation.py`)
Calculates performance metrics:
- Accuracy
- Precision  
- Recall
- F1-Score
- AUC-ROC (primary metric)

Generates confusion matrices and saves results to CSV.

### 4. Visualization (`src/visualization.py`)
Creates comprehensive visualizations:
- Performance comparison bar charts (combined and individual)
- Confusion matrices for best models (combined and individual)
- ROC curves for all models (combined and large individual)
- Data transformation impact analysis (combined and per-algorithm)
- Results summary table
- Individual readable figures for better report inclusion

## Expected Results

Actual performance on reduced dataset (10,000 samples):
- **Best:** Neural Network (Standardized) - AUC-ROC 0.797, Accuracy 72.4%
- **Second:** Neural Network (MinMax) - AUC-ROC 0.793, Accuracy 72.4%
- **Third:** Logistic Regression (Standardized) - AUC-ROC 0.792, Accuracy 73.1%
- **Key Finding:** Neural networks perform best on this reduced dataset
- **Demonstrates:** Data transformations significantly impact model performance

## Visualizations

**Combined Figures:**
- `model_comparison.png` - All metrics in one view
- `confusion_matrices.png` - Top 3 models side-by-side
- `roc_curves.png` - All ROC curves together
- `transformation_impact.png` - All algorithms' transformation effects

**Individual Figures (for reports):**
- `individual_accuracy_comparison.png` - Clear accuracy rankings
- `individual_precision_comparison.png` - Precision focus
- `individual_recall_comparison.png` - Recall analysis
- `individual_auc_roc_comparison.png` - Primary metric comparison
- `individual_confusion_matrix_*.png` - Separate matrices for each top model
- `individual_roc_curves_large.png` - Larger, clearer ROC plot
- `individual_transformation_*.png` - Per-algorithm transformation impact

## Dependencies

Key libraries (see `requirements.txt` for versions):
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms
- `matplotlib` + `seaborn` - Visualization

## Dataset

```
Cardiovascular Disease dataset. (2019). Kaggle.
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
```