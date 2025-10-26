import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os


# Set style for publication-quality plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results():
    """Load evaluation results."""
    return pd.read_csv('results/model_results.csv')


def plot_model_comparison(results_df, save_dir='figures'):
    """
    Create bar plots comparing all models across different metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC-ROC']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value
        data_sorted = results_df.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(data_sorted)), data_sorted[metric])
        
        # Color bars with gradient
        colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors_gradient):
            bar.set_color(color)
        
        # Customize plot
        ax.set_yticks(range(len(data_sorted)))
        ax.set_yticklabels(data_sorted['Model'], fontsize=10)
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.set_xlim([0.5, 0.85])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx_val, row) in enumerate(data_sorted.iterrows()):
            ax.text(row[metric] + 0.005, i, f'{row[metric]:.4f}', 
                   va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Performance Comparison Across All Models and Transformations', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/model_comparison.png")
    plt.close()


def plot_confusion_matrices(save_dir='figures'):
    """
    Create confusion matrices for the three best-performing models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test data
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    X_train_orig = pd.read_csv('data/X_train.csv')
    X_test_orig = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    # Prepare transformations
    scaler_std = StandardScaler()
    scaler_std.fit(X_train_orig)
    X_test_std = scaler_std.transform(X_test_orig)
    
    scaler_mm = MinMaxScaler()
    scaler_mm.fit(X_train_orig)
    X_test_mm = scaler_mm.transform(X_test_orig)
    
    # Models to plot (best from each algorithm)
    models_info = [
        ('svm', 'standardized', X_test_std, 'SVM (Standardized)'),
        ('neural_network', 'minmax', X_test_mm, 'Neural Network (MinMax)'),
        ('logistic_regression', 'standardized', X_test_std, 'Logistic Regression (Standardized)')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, trans, X_test, display_name) in enumerate(models_info):
        # Load model
        with open(f'models/{model_name}_{trans}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=['No CVD', 'CVD'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
        axes[idx].set_title(f'{display_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_ylabel('True Label' if idx == 0 else '', fontsize=11)
        
        # Add accuracy annotation
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                      ha='center', transform=axes[idx].transAxes,
                      fontsize=10, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - Best Models from Each Algorithm', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/confusion_matrices.png")
    plt.close()


def plot_roc_curves(save_dir='figures'):
    """
    Plot ROC curves for all models on the same plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test data
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    X_train_orig = pd.read_csv('data/X_train.csv')
    X_test_orig = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    # Prepare transformations
    scaler_std = StandardScaler()
    scaler_std.fit(X_train_orig)
    X_test_std = scaler_std.transform(X_test_orig)
    
    scaler_mm = MinMaxScaler()
    scaler_mm.fit(X_train_orig)
    X_test_mm = scaler_mm.transform(X_test_orig)
    
    # Models to plot
    models_info = [
        ('logistic_regression', 'original', X_test_orig, 'Logistic Regression (Original)', '#7F7F7F', '--'),
        ('logistic_regression', 'standardized', X_test_std, 'Logistic Regression (Standardized)', '#1F77B4', '-'),
        ('svm', 'original', X_test_orig, 'SVM (Original)', '#FF7F0E', ':'),
        ('svm', 'standardized', X_test_std, 'SVM (Standardized)', '#2CA02C', '-'),
        ('neural_network', 'standardized', X_test_std, 'Neural Network (Standardized)', '#D62728', '--'),
        ('neural_network', 'minmax', X_test_mm, 'Neural Network (MinMax)', '#9467BD', '-')
    ]
    
    plt.figure(figsize=(10, 8))
    
    for model_name, trans, X_test, display_name, color, linestyle in models_info:
        # Load model
        with open(f'models/{model_name}_{trans}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Get probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, linewidth=2, label=f'{display_name} (AUC = {roc_auc:.4f})',
                color=color, linestyle=linestyle)
    
    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.5000)')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - All Models and Transformations', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/roc_curves.png")
    plt.close()


def plot_transformation_impact(results_df, save_dir='figures'):
    """
    Visualize the impact of transformations on each algorithm.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract algorithm and transformation from model names
    results_df['Algorithm'] = results_df['Model'].str.extract(r'^([^\(]+)')[0].str.strip()
    results_df['Transformation'] = results_df['Model'].str.extract(r'\(([^\)]+)\)')[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    algorithms = ['Logistic Regression', 'SVM', 'Neural Network']
    colors = {'Original': '#E74C3C', 'Standardized': '#3498DB', 'MinMax': '#2ECC71'}
    
    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx]
        
        # Filter data for this algorithm
        algo_data = results_df[results_df['Algorithm'] == algorithm].copy()
        
        if len(algo_data) == 0:
            continue
        
        # Plot grouped bars
        transformations = algo_data['Transformation'].values
        auc_scores = algo_data['AUC-ROC'].values
        
        x_pos = np.arange(len(transformations))
        bars = ax.bar(x_pos, auc_scores, color=[colors.get(t, '#95A5A6') for t in transformations])
        
        # Customize
        ax.set_ylabel('AUC-ROC Score', fontsize=11, fontweight='bold')
        ax.set_title(algorithm, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(transformations, rotation=15, ha='right')
        ax.set_ylim([0.5, 0.85])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, auc_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.4f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Add improvement annotation for SVM
        if algorithm == 'SVM' and len(auc_scores) == 2:
            improvement = auc_scores[1] - auc_scores[0]
            improvement_pct = (improvement / auc_scores[0]) * 100
            ax.annotate(f'+{improvement:.4f}\n(+{improvement_pct:.1f}%)',
                       xy=(0.5, 0.7), xycoords='axes fraction',
                       fontsize=11, fontweight='bold', color='#27AE60',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='#27AE60', linewidth=2))
    
    plt.suptitle('Impact of Data Transformations on Model Performance', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/transformation_impact.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/transformation_impact.png")
    plt.close()


def create_summary_table(results_df, save_dir='figures'):
    """
    Create a summary table image of all results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Round values for display
    display_df = results_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(display_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ECF0F1')
            else:
                cell.set_facecolor('#FFFFFF')
    
    # Highlight best AUC-ROC
    best_idx = results_df['AUC-ROC'].idxmax()
    auc_col_idx = list(display_df.columns).index('AUC-ROC')
    cell = table[(best_idx + 1, auc_col_idx)]
    cell.set_facecolor('#27AE60')
    cell.set_text_props(weight='bold', color='white')
    
    plt.title('Complete Results Summary - All Models', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/results_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/results_table.png")
    plt.close()


def main():
    """
    Generate all visualizations for the report.
    """
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    results_df = load_results()
    print(f"✓ Loaded results for {len(results_df)} models")
    
    # Create plots
    print("\nCreating visualizations...")
    print("\n1. Model Comparison Bar Charts")
    plot_model_comparison(results_df)
    
    print("\n2. Confusion Matrices")
    plot_confusion_matrices()
    
    print("\n3. ROC Curves")
    plot_roc_curves()
    
    print("\n4. Transformation Impact Analysis")
    plot_transformation_impact(results_df)
    
    print("\n5. Results Summary Table")
    create_summary_table(results_df)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nAll figures saved to: figures/")
    print("\nGenerated files:")
    print("  • model_comparison.png")
    print("  • confusion_matrices.png")
    print("  • roc_curves.png")
    print("  • transformation_impact.png")
    print("  • results_table.png")


if __name__ == "__main__":
    main()