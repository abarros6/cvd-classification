"""
Visualization Components

Creates comprehensive visualizations for Assignment 2 deep learning tuning analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json


class AssignmentVisualizer:
    """Creates all required visualizations for Assignment 2."""
    
    def __init__(self, 
                 results_dir: str = "../results",
                 figures_dir: str = "../figures"):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing results
            figures_dir: Directory to save figures
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
    
    def plot_learning_curves(self, history_file: str = "training_history.csv") -> None:
        """Plot learning curves to show the network is learning."""
        print("Creating learning curves...")
        
        history_path = self.results_dir / history_file
        if not history_path.exists():
            print(f"Warning: Training history file not found: {history_path}")
            return
        
        # Load training history
        history = pd.read_csv(history_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Deep Neural Network Learning Curves', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['loss'], label='Training Loss', linewidth=2, alpha=0.8)
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Binary Crossentropy Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC curves
        axes[0, 1].plot(epochs, history['auc'], label='Training AUC-ROC', linewidth=2, alpha=0.8)
        axes[0, 1].plot(epochs, history['val_auc'], label='Validation AUC-ROC', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('AUC-ROC Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1, 0].plot(epochs, history['accuracy'], label='Training Accuracy', linewidth=2, alpha=0.8)
        axes[1, 0].plot(epochs, history['val_accuracy'], label='Validation Accuracy', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Accuracy Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate effect (if available)
        if 'lr' in history.columns:
            axes[1, 1].plot(epochs, history['lr'], label='Learning Rate', linewidth=2, alpha=0.8, color='orange')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show convergence analysis instead
            window_size = min(10, len(history) // 4)
            val_auc_smooth = history['val_auc'].rolling(window=window_size, center=True).mean()
            axes[1, 1].plot(epochs, history['val_auc'], alpha=0.3, color='blue', label='Validation AUC')
            axes[1, 1].plot(epochs, val_auc_smooth, linewidth=2, color='blue', label=f'Smoothed (window={window_size})')
            axes[1, 1].set_title('Convergence Analysis')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation AUC-ROC')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "learning_curves.png")
        plt.close()
        print(f"  Saved: learning_curves.png")
    
    def plot_architecture_comparison(self) -> None:
        """
        Plot architecture tuning results (Phase 1).
        Heatmap showing performance by layers × neurons.
        """
        print("Creating architecture comparison...")
        
        # Load Phase 1 results
        phase1_path = self.results_dir / "phase1_architecture_results.csv"
        if not phase1_path.exists():
            print(f"Warning: Phase 1 results not found: {phase1_path}")
            return
        
        results = pd.read_csv(phase1_path)
        
        # Create pivot table for heatmap
        pivot_data = results.pivot(
            index='num_layers',
            columns='neurons_per_layer', 
            values='val_auc'
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Architecture Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        # Heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            center=pivot_data.mean().mean(),
            ax=axes[0],
            cbar_kws={'label': 'Validation AUC-ROC'}
        )
        axes[0].set_title('Architecture Performance Matrix')
        axes[0].set_xlabel('Neurons per Layer')
        axes[0].set_ylabel('Number of Hidden Layers')
        
        # Bar plot showing best configuration for each depth
        depth_best = results.groupby('num_layers')['val_auc'].max().reset_index()
        bars = axes[1].bar(depth_best['num_layers'], depth_best['val_auc'], alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1].set_title('Best Performance by Network Depth')
        axes[1].set_xlabel('Number of Hidden Layers')
        axes[1].set_ylabel('Best Validation AUC-ROC')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, depth_best['val_auc']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "architecture_comparison.png")
        plt.close()
        print(f"  Saved: architecture_comparison.png")
    
    def plot_hyperparameter_effects(self) -> None:
        """
        Plot effects of different hyperparameters across all phases.
        """
        print("Creating hyperparameter effects visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Hyperparameter Effects on Model Performance', fontsize=16, fontweight='bold')
        
        # Phase 2 - Regularization effects
        phase2_path = self.results_dir / "phase2_regularization_results.csv"
        if phase2_path.exists():
            reg_results = pd.read_csv(phase2_path)
            
            # Dropout effect
            dropout_effect = reg_results.groupby('dropout_rate')['val_auc'].agg(['mean', 'std']).reset_index()
            axes[0, 0].errorbar(dropout_effect['dropout_rate'], dropout_effect['mean'], 
                              yerr=dropout_effect['std'], marker='o', capsize=5, capthick=2, linewidth=2)
            axes[0, 0].set_title('Dropout Rate Effect')
            axes[0, 0].set_xlabel('Dropout Rate')
            axes[0, 0].set_ylabel('Validation AUC-ROC')
            axes[0, 0].grid(True, alpha=0.3)
            
            # L2 regularization effect
            l2_effect = reg_results.groupby('l2_regularization')['val_auc'].agg(['mean', 'std']).reset_index()
            axes[0, 1].errorbar(l2_effect['l2_regularization'], l2_effect['mean'],
                              yerr=l2_effect['std'], marker='s', capsize=5, capthick=2, linewidth=2)
            axes[0, 1].set_title('L2 Regularization Effect')
            axes[0, 1].set_xlabel('L2 Regularization')
            axes[0, 1].set_ylabel('Validation AUC-ROC')
            axes[0, 1].set_xscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Phase 3 - Training optimization effects
        phase3_path = self.results_dir / "phase3_training_results.csv"
        if phase3_path.exists():
            train_results = pd.read_csv(phase3_path)
            
            # Learning rate effect (binned)
            lr_bins = pd.cut(train_results['learning_rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            lr_effect = train_results.groupby(lr_bins)['val_auc'].agg(['mean', 'std']).reset_index()
            lr_effect = lr_effect.dropna()
            
            x_pos = range(len(lr_effect))
            axes[0, 2].bar(x_pos, lr_effect['mean'], yerr=lr_effect['std'], 
                          capsize=5, alpha=0.7, color='lightcoral', edgecolor='darkred')
            axes[0, 2].set_title('Learning Rate Range Effect')
            axes[0, 2].set_xlabel('Learning Rate Range')
            axes[0, 2].set_ylabel('Validation AUC-ROC')
            axes[0, 2].set_xticks(x_pos)
            axes[0, 2].set_xticklabels(lr_effect['learning_rate'], rotation=45)
            axes[0, 2].grid(True, alpha=0.3, axis='y')
            
            # Batch size effect
            batch_effect = train_results.groupby('batch_size')['val_auc'].agg(['mean', 'std']).reset_index()
            axes[1, 0].bar(batch_effect['batch_size'], batch_effect['mean'], yerr=batch_effect['std'],
                          capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            axes[1, 0].set_title('Batch Size Effect')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Validation AUC-ROC')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Optimizer comparison
            opt_effect = train_results.groupby('optimizer')['val_auc'].agg(['mean', 'std']).reset_index()
            axes[1, 1].bar(opt_effect['optimizer'], opt_effect['mean'], yerr=opt_effect['std'],
                          capsize=5, alpha=0.7, color='plum', edgecolor='purple')
            axes[1, 1].set_title('Optimizer Comparison')
            axes[1, 1].set_xlabel('Optimizer')
            axes[1, 1].set_ylabel('Validation AUC-ROC')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Training time analysis (if available)
        all_results_path = self.results_dir / "complete_tuning_results.csv"
        if all_results_path.exists():
            all_results = pd.read_csv(all_results_path)
            if 'training_time' in all_results.columns:
                axes[1, 2].scatter(all_results['training_time'], all_results['val_auc'], 
                                 alpha=0.6, s=30, color='orange', edgecolor='darkorange')
                axes[1, 2].set_title('Performance vs Training Time')
                axes[1, 2].set_xlabel('Training Time (seconds)')
                axes[1, 2].set_ylabel('Validation AUC-ROC')
                axes[1, 2].grid(True, alpha=0.3)
        
        # Remove empty subplot if no data
        if not all_results_path.exists() or 'training_time' not in pd.read_csv(all_results_path).columns:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "hyperparameter_effects.png")
        plt.close()
        print(f"  Saved: hyperparameter_effects.png")
    
    def plot_final_comparison(self) -> None:
        """
        Plot final comparison between tuned model, baseline, and Assignment 1 models.
        Required: Must show comparisons with Assignment 1.
        """
        print("Creating final model comparison...")
        
        # Load comparison results
        comparison_path = self.results_dir / "comparison_results.csv"
        if not comparison_path.exists():
            print(f"Warning: Comparison results not found: {comparison_path}")
            return
        
        comparison_df = pd.read_csv(comparison_path)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison: Assignment 2 vs Assignment 1', 
                    fontsize=16, fontweight='bold')
        
        # Metrics to plot
        metrics = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Sort by the current metric
            sorted_df = comparison_df.sort_values(metric, ascending=True)
            
            # Create horizontal bar plot
            colors = ['red' if 'Assignment 2' in src else 'blue' for src in sorted_df['Source']]
            bars = ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels(sorted_df['Model'], fontsize=9)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
                ax.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center', fontweight='bold', fontsize=8)
            
            # Highlight Assignment 2 models
            for i, source in enumerate(sorted_df['Source']):
                if 'Assignment 2' in source:
                    bars[i].set_edgecolor('darkred')
                    bars[i].set_linewidth(2)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Assignment 2'),
            Patch(facecolor='blue', alpha=0.7, label='Assignment 1')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "final_comparison.png")
        plt.close()
        print(f"  Saved: final_comparison.png")
    
    def plot_tuning_summary(self) -> None:
        """
        Create a comprehensive tuning summary visualization.
        """
        print("Creating tuning summary...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Hyperparameter Tuning Summary', fontsize=16, fontweight='bold')
        
        # Load all phase results
        phase_results = {}
        phase_files = {
            1: "phase1_architecture_results.csv",
            2: "phase2_regularization_results.csv", 
            3: "phase3_training_results.csv"
        }
        
        for phase, filename in phase_files.items():
            file_path = self.results_dir / filename
            if file_path.exists():
                phase_results[phase] = pd.read_csv(file_path)
        
        # Plot 1: Performance progression through phases
        if phase_results:
            phase_best = []
            phase_names = []
            for phase, df in phase_results.items():
                phase_best.append(df['val_auc'].max())
                phase_names.append(f'Phase {phase}')
            
            axes[0, 0].plot(range(len(phase_best)), phase_best, 'o-', linewidth=3, markersize=8, color='green')
            axes[0, 0].set_title('Best Performance by Phase')
            axes[0, 0].set_ylabel('Best Validation AUC-ROC')
            axes[0, 0].set_xticks(range(len(phase_names)))
            axes[0, 0].set_xticklabels(phase_names)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add improvement annotations
            for i in range(1, len(phase_best)):
                improvement = phase_best[i] - phase_best[i-1]
                axes[0, 0].annotate(f'+{improvement:.4f}', 
                                  xy=(i, phase_best[i]), 
                                  xytext=(i, phase_best[i] + 0.002),
                                  ha='center', fontweight='bold', color='darkgreen')
        
        # Plot 2: Experiment distribution by phase
        if phase_results:
            phase_counts = [len(df) for df in phase_results.values()]
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            bars = axes[0, 1].bar(range(len(phase_counts)), phase_counts, color=colors, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Experiments per Phase')
            axes[0, 1].set_ylabel('Number of Experiments')
            axes[0, 1].set_xticks(range(len(phase_names)))
            axes[0, 1].set_xticklabels(phase_names)
            
            # Add count labels
            for bar, count in zip(bars, phase_counts):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Performance distribution across all experiments
        all_results_path = self.results_dir / "complete_tuning_results.csv"
        if all_results_path.exists():
            all_results = pd.read_csv(all_results_path)
            axes[0, 2].hist(all_results['val_auc'], bins=15, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 2].axvline(all_results['val_auc'].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {all_results["val_auc"].mean():.4f}')
            axes[0, 2].axvline(all_results['val_auc'].max(), color='green', linestyle='--',
                              linewidth=2, label=f'Best: {all_results["val_auc"].max():.4f}')
            axes[0, 2].set_title('Performance Distribution')
            axes[0, 2].set_xlabel('Validation AUC-ROC')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plots 4-6: Top configurations from each phase
        for i, (phase, df) in enumerate(phase_results.items()):
            if i >= 3:  # Only plot first 3 phases
                break
            
            ax = axes[1, i]
            top_configs = df.nlargest(5, 'val_auc')
            
            y_pos = range(len(top_configs))
            bars = ax.barh(y_pos, top_configs['val_auc'], alpha=0.7, color=colors[i], edgecolor='black')
            
            ax.set_title(f'Phase {phase} - Top 5 Configurations')
            ax.set_xlabel('Validation AUC-ROC')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'Config {j+1}' for j in range(len(top_configs))])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, value in zip(bars, top_configs['val_auc']):
                ax.text(value + 0.0005, bar.get_y() + bar.get_height()/2,
                       f'{value:.4f}', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "tuning_summary.png")
        plt.close()
        print(f"  Saved: tuning_summary.png")
    
    def create_all_visualizations(self) -> None:
        """
        Create all required visualizations for Assignment 2.
        """
        print("=" * 60)
        print("CREATING ALL VISUALIZATIONS")
        print("=" * 60)
        
        # Required visualizations
        self.plot_learning_curves()           # Shows NN is learning
        self.plot_architecture_comparison()   # Phase 1 results
        self.plot_hyperparameter_effects()    # All hyperparameter effects
        self.plot_final_comparison()          # Comparison with Assignment 1
        
        # Additional analysis visualizations
        self.plot_tuning_summary()           # Overall tuning summary
        
        print("\n" + "=" * 60)
        print("ALL VISUALIZATIONS COMPLETED")
        print("=" * 60)
        print(f"Figures saved to: {self.figures_dir}")
        print("Generated files:")
        for fig_file in sorted(self.figures_dir.glob("*.png")):
            print(f"  - {fig_file.name}")


# Utility function for custom plots
def create_custom_comparison_plot(data: Dict[str, List[float]], 
                                title: str,
                                output_path: str) -> None:
    """
    Create custom comparison plot for specific analysis.
    
    Args:
        data: Dictionary with model names as keys and metric lists as values
        title: Plot title
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(next(iter(data.values()))))
    width = 0.15
    
    for i, (model, values) in enumerate(data.items()):
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(data) - 1) / 2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()