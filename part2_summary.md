# Assignment 2 Summary: Deep Neural Network Tuning

## Assignment Overview

**Assignment:** ECE 9603 Data Analytics Foundations - Assignment 2: Deep Learning Tuning  
**Building On:** Assignment 1 (CVD Classification)  
**Deadline:** [TBD]  
**Focus:** Build and extensively tune a deep neural network

---

## Assignment Requirements

### 1. Problem & Dataset Selection

**DECISION: Using the same CVD dataset from Assignment 1**

✅ **Allowed:** Can use same problem and dataset  
✅ **Our dataset:** Has 11 features (sufficient - requirement is "several independent variables")  
✅ **No feature engineering needed:** Already have enough features  
⚠️ **Cannot use:** Same dataset as group project

**Dataset Summary:**
- **Source:** Kaggle Cardiovascular Disease Dataset
- **Size:** 65,439 records after cleaning
- **Features:** 11 (age, gender, height, weight, bmi, bp_systolic, bp_diastolic, cholesterol, glucose, smoking, alcohol, physical_activity)
- **Target:** Binary (CVD present/absent)
- **Balance:** Perfect 50/50
- **Split:** 80/20 stratified (same as Assignment 1)

---

### 2. Architecture Selection

**REQUIREMENT:** Select ONE deep learning architecture and justify it

**Must Explain:**
- Which architectures were considered
- Why you selected the specific architecture
- Overview of selected architecture
- Important hyperparameters
- Which hyperparameters will be tuned

**Suggested Architecture for CVD Classification:**

**SELECTED: Deep Feedforward Neural Network (Deep MLP)**

**Architectures Considered:**
1. ✅ **Deep MLP** - SELECTED
   - Best for tabular healthcare data
   - Can handle 11 diverse features
   - Proven effective in Assignment 1
   - Flexible for deep architectures (5+ layers)

2. ❌ **Convolutional Neural Network (CNN)**
   - Rejected: Designed for spatial data (images)
   - Not suitable for tabular medical records

3. ❌ **Recurrent Neural Network (RNN/LSTM)**
   - Rejected: Designed for sequential/time-series data
   - Our data is cross-sectional (not sequential)

4. ❌ **Transformer**
   - Rejected: Overkill for 11 features
   - Requires large datasets (we have 65k, need 100k+)

**Justification for Deep MLP:**
- Tabular data with mixed feature types (continuous + categorical)
- Proven baseline from Assignment 1 (0.797 AUC-ROC)
- Scalable to deep architectures (can add many layers)
- Well-suited for non-linear medical risk relationships

---

### 3. Hyperparameter Tuning Requirements

**MINIMUM REQUIREMENTS:**
1. ✅ Number of layers (test 3, 5, 7, 10 layers)
2. ✅ Number of neurons per layer (test 32, 64, 128, 256)
3. ✅ Two additional hyperparameters (e.g., learning rate, dropout)
4. ✅ At least one network with 5+ hidden layers

**SUGGESTED HYPERPARAMETERS TO TUNE:**

**Critical (MUST tune):**
1. **Number of hidden layers** - [3, 5, 7, 10, 12]
2. **Neurons per layer** - [32, 64, 128, 256, 512]
3. **Learning rate** - [0.0001, 0.001, 0.01, 0.1]
4. **Dropout rate** - [0.0, 0.2, 0.3, 0.5]

**Optional (SHOULD consider):**
5. **Batch size** - [32, 64, 128, 256]
6. **Optimizer** - [Adam, SGD, RMSprop]
7. **Activation function** - [ReLU, LeakyReLU, ELU]
8. **L2 regularization (weight decay)** - [0, 0.0001, 0.001, 0.01]
9. **Batch normalization** - [True, False]

**Architecture-specific:**
10. **Layer size pattern** - [constant, decreasing, pyramid, hourglass]
    - Constant: [128, 128, 128, 128]
    - Decreasing: [256, 128, 64, 32]
    - Pyramid: [32, 64, 128, 64, 32]

---

### 4. Tuning Strategy

**REQUIREMENT:** Explain tuning strategy in detail (enough to recreate)

**SUGGESTED APPROACH:**

**Phase 1: Coarse Grid Search (Architecture)**
- Fix: learning_rate=0.001, dropout=0.3, batch_size=64
- Tune: num_layers, neurons_per_layer
- Strategy: Grid search over {3,5,7,10} layers × {64,128,256} neurons
- Metric: Validation AUC-ROC
- Goal: Find best architecture depth and width

**Phase 2: Fine-tune Regularization**
- Fix: Best architecture from Phase 1
- Tune: dropout_rate, l2_regularization
- Strategy: Grid search over dropout {0.0, 0.2, 0.4} × L2 {0, 0.001, 0.01}
- Metric: Validation AUC-ROC
- Goal: Prevent overfitting

**Phase 3: Optimize Training**
- Fix: Best architecture + regularization
- Tune: learning_rate, batch_size, optimizer
- Strategy: Random search (20 trials)
- Ranges:
  - learning_rate: log-uniform [1e-4, 1e-1]
  - batch_size: choice [32, 64, 128, 256]
  - optimizer: choice [Adam, RMSprop]
- Metric: Validation AUC-ROC
- Goal: Maximize convergence speed and final performance

**Validation Strategy:**
- Use same 80/20 split as Assignment 1
- Further split training data: 90% train / 10% validation
- Final test on held-out 20% (same as Assignment 1)
- Early stopping: patience=20 epochs, monitor validation loss

---

### 5. Evaluation & Comparison Requirements

**MUST COMPARE:**
1. ✅ Tuned model vs Untuned model
2. ✅ Tuned deep NN vs Assignment 1 models (Logistic Regression, SVM, shallow NN)
3. ✅ Show learning curves (demonstrate the NN is learning)

**MUST INCLUDE:**
- 2+ metrics for comparison (suggest: AUC-ROC, Accuracy, F1-Score)
- Graphs comparing tuning experiments
- Training/validation loss and accuracy curves
- Analysis and insights

---

## Implementation Plan

### File Structure (New Files for Assignment 2)
```
cvd-classification-assignment/
├── assignment2/                   # NEW: Assignment 2 code
│   ├── __init__.py
│   ├── deep_nn_config.py         # Hyperparameter configurations
│   ├── deep_nn_model.py          # Deep NN architecture builder
│   ├── hyperparameter_tuning.py  # Grid/random search implementation
│   ├── training.py               # Training loop with early stopping
│   ├── evaluation.py             # Evaluation & comparison
│   └── visualization.py          # Learning curves, comparison plots
├── results_assignment2/           # NEW: Assignment 2 results
│   ├── tuning_results.csv        # All tuning experiments
│   ├── best_model_config.json    # Best hyperparameters
│   └── comparison_results.csv    # Tuned vs Assignment 1
├── figures_assignment2/           # NEW: Assignment 2 figures
│   ├── architecture_comparison.png
│   ├── learning_curves.png
│   ├── tuning_heatmap.png
│   ├── final_comparison.png
│   └── training_history.png
├── models_assignment2/            # NEW: Assignment 2 models
│   ├── untuned_baseline.h5
│   ├── best_tuned_model.h5
│   └── tuning_checkpoints/
├── data/                          # REUSE from Assignment 1
│   ├── cardio_train.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── src/                           # KEEP Assignment 1 code
│   └── [Assignment 1 files]
├── REPORT_ASSIGNMENT2.md          # NEW: Assignment 2 report
└── run_assignment2.py             # NEW: Master script
```

---

## Detailed Implementation Steps

### Step 1: Setup Deep NN Framework

**Use TensorFlow/Keras (more flexible than scikit-learn for deep networks)**
```python
# deep_nn_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_deep_nn(
    input_dim=11,
    num_layers=5,
    neurons_per_layer=128,
    dropout_rate=0.3,
    l2_reg=0.001,
    activation='relu',
    use_batch_norm=False
):
    """
    Build a deep feedforward neural network.
    
    Architecture patterns supported:
    - Constant: Same neurons in all layers
    - Decreasing: Gradually reduce neurons
    - Pyramid: Increase then decrease
    """
    
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i in range(num_layers):
        # Calculate neurons for this layer (if using pattern)
        neurons = calculate_layer_neurons(neurons_per_layer, i, num_layers)
        
        # Dense layer with L2 regularization
        model.add(layers.Dense(
            neurons,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        
        # Optional batch normalization
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        
        # Dropout for regularization
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model
```

### Step 2: Implement Tuning Strategy
```python
# hyperparameter_tuning.py

def grid_search_architecture(X_train, y_train, X_val, y_val):
    """
    Phase 1: Grid search over architecture
    """
    results = []
    
    layer_configs = [3, 5, 7, 10]
    neuron_configs = [64, 128, 256]
    
    for num_layers in layer_configs:
        for neurons in neuron_configs:
            print(f"Testing: {num_layers} layers, {neurons} neurons")
            
            # Build model
            model = build_deep_nn(
                num_layers=num_layers,
                neurons_per_layer=neurons,
                dropout_rate=0.3,
                l2_reg=0.001
            )
            
            # Compile
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            
            # Train with early stopping
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_auc',
                        patience=20,
                        restore_best_weights=True,
                        mode='max'
                    )
                ],
                verbose=0
            )
            
            # Evaluate
            val_auc = max(history.history['val_auc'])
            
            results.append({
                'num_layers': num_layers,
                'neurons': neurons,
                'val_auc': val_auc,
                'best_epoch': len(history.history['val_auc'])
            })
    
    return pd.DataFrame(results)


def random_search_optimization(X_train, y_train, X_val, y_val, best_arch):
    """
    Phase 3: Random search over training hyperparameters
    """
    results = []
    
    for trial in range(20):
        # Sample hyperparameters
        lr = np.random.loguniform(1e-4, 1e-1)
        batch_size = np.random.choice([32, 64, 128, 256])
        optimizer_name = np.random.choice(['adam', 'rmsprop'])
        
        # Build model with best architecture
        model = build_deep_nn(**best_arch)
        
        # Select optimizer
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        
        # Compile and train
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stopping_callback],
            verbose=0
        )
        
        # Record results
        val_auc = max(history.history['val_auc'])
        
        results.append({
            'trial': trial,
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'val_auc': val_auc
        })
    
    return pd.DataFrame(results)
```

### Step 3: Training & Evaluation
```python
# training.py

def train_final_model(X_train, y_train, X_val, y_val, best_config):
    """
    Train final model with best hyperparameters
    """
    model = build_deep_nn(**best_config)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_config['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            'models_assignment2/best_tuned_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.CSVLogger('results_assignment2/training_history.csv')
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=best_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def compare_with_assignment1(tuned_model, X_test, y_test):
    """
    Compare tuned deep NN with Assignment 1 models
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    # Tuned deep NN predictions
    y_pred_proba = tuned_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    results = {
        'Tuned Deep NN': {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred)
        }
    }
    
    # Load Assignment 1 results
    assignment1_results = pd.read_csv('../results/model_results.csv')
    
    # Add to comparison
    for _, row in assignment1_results.iterrows():
        results[row['Model']] = {
            'accuracy': row['Accuracy'],
            'auc_roc': row['AUC-ROC'],
            'f1_score': row['F1-Score']
        }
    
    return pd.DataFrame(results).T
```

### Step 4: Visualization
```python
# visualization.py

def plot_learning_curves(history):
    """
    Plot training and validation loss/accuracy over epochs
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Crossentropy Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # AUC
    axes[1].plot(history.history['auc'], label='Training AUC-ROC')
    axes[1].plot(history.history['val_auc'], label='Validation AUC-ROC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('Training vs Validation AUC-ROC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_assignment2/learning_curves.png', dpi=300)


def plot_architecture_comparison(tuning_results):
    """
    Heatmap showing performance by architecture
    """
    pivot = tuning_results.pivot(
        index='num_layers',
        columns='neurons',
        values='val_auc'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis')
    plt.xlabel('Neurons per Layer')
    plt.ylabel('Number of Hidden Layers')
    plt.title('Architecture Tuning Results (Validation AUC-ROC)')
    plt.tight_layout()
    plt.savefig('figures_assignment2/architecture_comparison.png', dpi=300)


def plot_final_comparison(comparison_df):
    """
    Bar chart comparing tuned NN with all Assignment 1 models
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['accuracy', 'auc_roc', 'f1_score']
    titles = ['Accuracy', 'AUC-ROC', 'F1-Score']
    
    for ax, metric, title in zip(axes, metrics, titles):
        comparison_df[metric].plot(kind='barh', ax=ax)
        ax.set_xlabel(title)
        ax.set_title(f'Model Comparison - {title}')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_assignment2/final_comparison.png', dpi=300)
```

---

## Expected Results Structure

### Tuning Results Table:

| Experiment | Layers | Neurons | Dropout | L2 Reg | LR | Batch Size | Val AUC | Test AUC |
|------------|--------|---------|---------|--------|----|-----------|---------| ---------|
| Baseline (untuned) | 2 | 64,32 | 0.0 | 0.0 | 0.001 | 32 | 0.793 | 0.797 |
| Exp 1 | 3 | 64 | 0.3 | 0.001 | 0.001 | 64 | 0.801 | - |
| Exp 2 | 5 | 128 | 0.3 | 0.001 | 0.001 | 64 | 0.812 | - |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Best | 7 | 128 | 0.3 | 0.001 | 0.0005 | 64 | 0.825 | 0.822 |

### Final Comparison Table:

| Model | Accuracy | AUC-ROC | F1-Score |
|-------|----------|---------|----------|
| **Tuned Deep NN** | **0.745** | **0.822** | **0.740** |
| Untuned Deep NN | 0.724 | 0.797 | 0.718 |
| Neural Network (A1) | 0.724 | 0.797 | 0.718 |
| Logistic Regression (A1) | 0.731 | 0.792 | 0.715 |
| SVM (A1) | 0.725 | 0.791 | 0.714 |

**Expected improvement:** 2-3% AUC-ROC improvement from tuning

---

## Report Structure for Assignment 2

### 1. Problem Description and Data (2 points)
- **Reuse from Assignment 1** (if no issues)
- Brief reminder of CVD classification problem
- Dataset description (can reference Assignment 1)
- Mention: "Using same dataset as Assignment 1 per course guidelines"

### 2. Background (4 points)
**Must include:**
- Deep learning overview for tabular data
- Why Deep MLP is appropriate for CVD classification
- Architectures considered and why Deep MLP was selected
- Detailed explanation of architecture:
  - Forward propagation
  - Backpropagation
  - Loss function (binary crossentropy)
  - Optimization (Adam, RMSprop)
- Hyperparameters explained:
  - **Number of layers:** Controls model capacity and depth
  - **Neurons per layer:** Controls width and representation power
  - **Learning rate:** Controls optimization step size
  - **Dropout:** Prevents overfitting by randomly dropping neurons
  - **L2 regularization:** Penalizes large weights
  - **Batch size:** Affects gradient estimation and convergence
- Which hyperparameters you tuned and why

### 3. Methodology (7 points)
**Must include:**
- Data preprocessing (same as Assignment 1: standardization)
- Train/validation/test split strategy:
  - Test: 20% (same as Assignment 1)
  - Train: 72% (90% of remaining 80%)
  - Validation: 8% (10% of remaining 80%)
- Tuning strategy (DETAILED):
  - Phase 1: Architecture grid search
  - Phase 2: Regularization grid search
  - Phase 3: Training hyperparameters random search
  - Hyperparameter ranges tested
  - Number of experiments run
  - Validation strategy
  - Early stopping criteria
- Model training procedure
- Evaluation metrics (AUC-ROC, Accuracy, F1-Score)

### 4. Results (7 points)
**Must include:**
- **Tuning results:**
  - Table of all experiments
  - Heatmap/graph showing architecture performance
  - Best hyperparameters identified
- **Learning demonstration:**
  - Training/validation loss curves
  - Training/validation AUC curves
  - Show convergence (proves NN is learning)
- **Comparison with untuned:**
  - Metrics table (tuned vs untuned)
  - Improvement quantified
- **Comparison with Assignment 1:**
  - All 3 metrics compared
  - Bar charts showing improvements
- **Analysis:**
  - Why tuning improved performance
  - Which hyperparameters had most impact
  - Clinical implications of improvements
  - Discussion of overfitting/underfitting observations

---

## Key Differences from Assignment 1

| Aspect | Assignment 1 | Assignment 2 |
|--------|-------------|-------------|
| **Focus** | Compare algorithms | Tune one deep architecture |
| **Hyperparameters** | Default only | Extensive tuning required |
| **NN Depth** | 2 layers (shallow) | 5-12 layers (deep) |
| **Framework** | Scikit-learn MLPClassifier | TensorFlow/Keras (more flexible) |
| **Validation** | Hold-out only | Hold-out + validation split for tuning |
| **Transformations** | Compare 3 types | Use best from Assignment 1 (standardization) |
| **Comparison** | 3 algorithms | Tuned vs untuned vs Assignment 1 |
| **Deliverable** | Algorithm comparison | Tuning methodology + improvement demonstration |

---

## Dependencies Update

Add to `requirements.txt`:
```txt
# Assignment 2 additions
tensorflow==2.15.0
keras==2.15.0
scikit-optimize==0.9.0  # For advanced hyperparameter search
```

---

## Timeline & Effort Estimation

### Phase 1: Setup (2-3 hours)
- Create new file structure
- Implement deep NN builder
- Set up data pipeline with validation split

### Phase 2: Architecture Tuning (3-4 hours)
- Implement grid search over layers/neurons
- Run experiments (12-20 configurations)
- Analyze results and select best architecture

### Phase 3: Regularization Tuning (2-3 hours)
- Grid search over dropout/L2
- Run experiments (9 configurations)
- Identify best regularization strategy

### Phase 4: Training Optimization (2-3 hours)
- Random search over LR/batch size/optimizer
- Run 20 random trials
- Select final hyperparameters

### Phase 5: Final Training & Evaluation (2-3 hours)
- Train best model on full training set
- Evaluate on test set
- Generate all comparison plots

### Phase 6: Reporting (4-5 hours)
- Write methodology section (detailed tuning strategy)
- Create results section with all plots
- Write analysis and discussion

**Total Estimated Time: 15-21 hours**

---

## Critical Success Factors

### MUST DO:
✅ At least one architecture with 5+ hidden layers  
✅ Tune minimum 4 hyperparameters (layers, neurons, +2 more)  
✅ Show learning curves (demonstrate NN is learning)  
✅ Compare: tuned vs untuned vs Assignment 1  
✅ Use 2+ metrics in comparisons  
✅ Provide detailed tuning methodology (reproducible)

### NICE TO HAVE:
- Automated hyperparameter search (Keras Tuner)
- Cross-validation within tuning process
- Ensemble of top-N models
- Feature importance analysis
- Error analysis by patient subgroups

---

## Questions to Answer in Report

1. **Why Deep MLP over other architectures?**
   → Tabular data, mixed features, proven baseline

2. **How did you decide on hyperparameter ranges?**
   → Literature review + Assignment 1 baseline + computational constraints

3. **What tuning strategy did you use and why?**
   → Multi-phase: grid (architecture) → grid (regularization) → random (optimization)

4. **How much improvement did tuning provide?**
   → Expected: 2-3% AUC-ROC improvement (0.797 → 0.820+)

5. **Which hyperparameters had the most impact?**
   → Likely: number of layers, learning rate, dropout

6. **Did you observe overfitting? How did you address it?**
   → Monitor val vs train, use dropout + L2 reg + early stopping

7. **Is the deeper network better than Assignment 1 shallow network?**
   → Should be yes, but diminishing returns after 7-10 layers

8. **What are the clinical implications?**
   → Higher AUC = better patient risk stratification

---

## Common Pitfalls to Avoid

❌ **Not showing learning curves** → Proves NN isn't just memorizing  
❌ **Only comparing to untuned** → Must also compare to Assignment 1  
❌ **Insufficient tuning detail** → Must be reproducible  
❌ **Forgetting 5+ layer requirement** → At least one deep architecture  
❌ **No validation split during tuning** → Would overfit to test set  
❌ **Tuning on test set** → Major methodological error  
❌ **Not explaining hyperparameter choices** → Need justification

---

## Success Metrics

**Minimum Success:**
- Tuned model improves over untuned by 1-2% AUC-ROC
- Tuned model matches or exceeds Assignment 1 best (0.797)
- Clear learning curves showing convergence
- Detailed methodology allowing reproduction

**Excellent Success:**
- Tuned model achieves 0.82+ AUC-ROC (3%+ improvement)
- Identifies clear hyperparameter importance ranking
- Provides clinical insights from improvements
- Demonstrates overfitting prevention effectively

---

**This assignment builds directly on Assignment 1 infrastructure. Reuse data pipeline, focus effort on deep architecture exploration and systematic hyperparameter optimization.**