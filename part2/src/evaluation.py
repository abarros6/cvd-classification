"""
Evaluation and Analysis

Comprehensive evaluation tools for deep neural network models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, results_dir: str = "../results"):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
    
    def analyze_tuning_results(self) -> Dict[str, Any]:
        """
        Analyze hyperparameter tuning results across all phases.
        
        Returns:
            Dictionary with analysis insights
        """
        print("=" * 60)
        print("ANALYZING HYPERPARAMETER TUNING RESULTS")
        print("=" * 60)
        
        analysis = {}
        
        # Load all phase results
        phase_files = [
            "phase1_architecture_results.csv",
            "phase2_regularization_results.csv", 
            "phase3_training_results.csv"
        ]
        
        all_results = []
        for file in phase_files:
            file_path = self.results_dir / file
            if file_path.exists():
                df = pd.read_csv(file_path)
                all_results.append(df)
        
        if not all_results:
            print("Warning: No tuning results found")
            return analysis
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Overall analysis
        analysis['total_experiments'] = len(combined_results)
        analysis['best_val_auc'] = combined_results['val_auc'].max()
        analysis['best_config_idx'] = combined_results['val_auc'].idxmax()
        
        print(f"Total experiments run: {analysis['total_experiments']}")
        print(f"Best validation AUC: {analysis['best_val_auc']:.4f}")
        
        # Phase 1 analysis (Architecture)
        if len(all_results) > 0:
            phase1_results = all_results[0]
            analysis['phase1'] = self._analyze_architecture_results(phase1_results)
        
        # Phase 2 analysis (Regularization) 
        if len(all_results) > 1:
            phase2_results = all_results[1]
            analysis['phase2'] = self._analyze_regularization_results(phase2_results)
        
        # Phase 3 analysis (Training)
        if len(all_results) > 2:
            phase3_results = all_results[2]
            analysis['phase3'] = self._analyze_training_results(phase3_results)
        
        # Hyperparameter importance
        analysis['hyperparameter_importance'] = self._calculate_hyperparameter_importance(combined_results)
        
        # Save analysis
        self._save_analysis(analysis, "tuning_analysis.json")
        
        return analysis
    
    def _analyze_architecture_results(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Phase 1 architecture search results."""
        print("\nPhase 1 - Architecture Analysis:")
        
        analysis = {}
        
        # Best architecture
        best_idx = results['val_auc'].idxmax()
        best_layers = results.loc[best_idx, 'num_layers']
        best_neurons = results.loc[best_idx, 'neurons_per_layer']
        best_auc = results.loc[best_idx, 'val_auc']
        
        analysis['best_architecture'] = {
            'num_layers': int(best_layers),
            'neurons_per_layer': int(best_neurons),
            'val_auc': float(best_auc)
        }
        
        print(f"  Best: {best_layers} layers, {best_neurons} neurons → AUC: {best_auc:.4f}")
        
        # Layer depth analysis
        layer_performance = results.groupby('num_layers')['val_auc'].agg(['mean', 'max', 'std'])
        analysis['layer_depth_analysis'] = {
            'mean_performance_by_depth': layer_performance['mean'].to_dict(),
            'best_performance_by_depth': layer_performance['max'].to_dict()
        }
        
        print("  Performance by depth:")
        for layers in sorted(layer_performance.index):
            mean_auc = layer_performance.loc[layers, 'mean']
            max_auc = layer_performance.loc[layers, 'max']
            print(f"    {layers} layers: mean={mean_auc:.4f}, max={max_auc:.4f}")
        
        # Neuron width analysis  
        neuron_performance = results.groupby('neurons_per_layer')['val_auc'].agg(['mean', 'max', 'std'])
        analysis['neuron_width_analysis'] = {
            'mean_performance_by_width': neuron_performance['mean'].to_dict(),
            'best_performance_by_width': neuron_performance['max'].to_dict()
        }
        
        print("  Performance by width:")
        for neurons in sorted(neuron_performance.index):
            mean_auc = neuron_performance.loc[neurons, 'mean']
            max_auc = neuron_performance.loc[neurons, 'max']
            print(f"    {neurons} neurons: mean={mean_auc:.4f}, max={max_auc:.4f}")
        
        return analysis
    
    def _analyze_regularization_results(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Phase 2 regularization search results."""
        print("\nPhase 2 - Regularization Analysis:")
        
        analysis = {}
        
        # Best regularization
        best_idx = results['val_auc'].idxmax()
        best_dropout = results.loc[best_idx, 'dropout_rate']
        best_l2 = results.loc[best_idx, 'l2_regularization']
        best_auc = results.loc[best_idx, 'val_auc']
        
        analysis['best_regularization'] = {
            'dropout_rate': float(best_dropout),
            'l2_regularization': float(best_l2),
            'val_auc': float(best_auc)
        }
        
        print(f"  Best: dropout={best_dropout}, l2={best_l2} → AUC: {best_auc:.4f}")
        
        # Dropout analysis
        dropout_performance = results.groupby('dropout_rate')['val_auc'].agg(['mean', 'max'])
        analysis['dropout_analysis'] = {
            'mean_performance_by_dropout': dropout_performance['mean'].to_dict(),
            'best_performance_by_dropout': dropout_performance['max'].to_dict()
        }
        
        print("  Performance by dropout:")
        for dropout in sorted(dropout_performance.index):
            mean_auc = dropout_performance.loc[dropout, 'mean']
            max_auc = dropout_performance.loc[dropout, 'max']
            print(f"    {dropout}: mean={mean_auc:.4f}, max={max_auc:.4f}")
        
        # L2 regularization analysis
        l2_performance = results.groupby('l2_regularization')['val_auc'].agg(['mean', 'max'])
        analysis['l2_analysis'] = {
            'mean_performance_by_l2': l2_performance['mean'].to_dict(),
            'best_performance_by_l2': l2_performance['max'].to_dict()
        }
        
        print("  Performance by L2 regularization:")
        for l2_reg in sorted(l2_performance.index):
            mean_auc = l2_performance.loc[l2_reg, 'mean']
            max_auc = l2_performance.loc[l2_reg, 'max']
            print(f"    {l2_reg}: mean={mean_auc:.4f}, max={max_auc:.4f}")
        
        return analysis
    
    def _analyze_training_results(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Phase 3 training optimization results."""
        print("\nPhase 3 - Training Optimization Analysis:")
        
        analysis = {}
        
        # Best training config
        best_idx = results['val_auc'].idxmax()
        best_lr = results.loc[best_idx, 'learning_rate']
        best_batch = results.loc[best_idx, 'batch_size']
        best_opt = results.loc[best_idx, 'optimizer']
        best_auc = results.loc[best_idx, 'val_auc']
        
        analysis['best_training'] = {
            'learning_rate': float(best_lr),
            'batch_size': int(best_batch),
            'optimizer': str(best_opt),
            'val_auc': float(best_auc)
        }
        
        print(f"  Best: lr={best_lr:.4f}, batch={best_batch}, opt={best_opt} → AUC: {best_auc:.4f}")
        
        # Learning rate analysis
        lr_bins = pd.cut(results['learning_rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        lr_performance = results.groupby(lr_bins)['val_auc'].agg(['mean', 'max', 'count'])
        
        print("  Performance by learning rate range:")
        for lr_range in lr_performance.index:
            if pd.notna(lr_range):
                mean_auc = lr_performance.loc[lr_range, 'mean']
                max_auc = lr_performance.loc[lr_range, 'max']
                count = lr_performance.loc[lr_range, 'count']
                print(f"    {lr_range}: mean={mean_auc:.4f}, max={max_auc:.4f} (n={count})")
        
        # Batch size analysis
        batch_performance = results.groupby('batch_size')['val_auc'].agg(['mean', 'max', 'count'])
        analysis['batch_size_analysis'] = {
            'mean_performance_by_batch': batch_performance['mean'].to_dict(),
            'best_performance_by_batch': batch_performance['max'].to_dict()
        }
        
        print("  Performance by batch size:")
        for batch_size in sorted(batch_performance.index):
            mean_auc = batch_performance.loc[batch_size, 'mean']
            max_auc = batch_performance.loc[batch_size, 'max']
            count = batch_performance.loc[batch_size, 'count']
            print(f"    {batch_size}: mean={mean_auc:.4f}, max={max_auc:.4f} (n={count})")
        
        # Optimizer analysis
        opt_performance = results.groupby('optimizer')['val_auc'].agg(['mean', 'max', 'count'])
        analysis['optimizer_analysis'] = {
            'mean_performance_by_optimizer': opt_performance['mean'].to_dict(),
            'best_performance_by_optimizer': opt_performance['max'].to_dict()
        }
        
        print("  Performance by optimizer:")
        for optimizer in opt_performance.index:
            mean_auc = opt_performance.loc[optimizer, 'mean']
            max_auc = opt_performance.loc[optimizer, 'max']
            count = opt_performance.loc[optimizer, 'count']
            print(f"    {optimizer}: mean={mean_auc:.4f}, max={max_auc:.4f} (n={count})")
        
        return analysis
    
    def _calculate_hyperparameter_importance(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate relative importance of different hyperparameters."""
        importance = {}
        
        # Identify numeric columns that are hyperparameters
        numeric_columns = ['num_layers', 'neurons_per_layer', 'dropout_rate', 
                          'l2_regularization', 'learning_rate', 'batch_size']
        
        for col in numeric_columns:
            if col in results.columns:
                # Calculate correlation with validation AUC
                correlation = results[col].corr(results['val_auc'])
                importance[col] = abs(correlation) if not pd.isna(correlation) else 0.0
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        print("\nHyperparameter Importance (correlation with validation AUC):")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {score:.3f}")
        
        return importance
    
    def analyze_learning_curves(self, history_file: str = "training_history.csv") -> Dict[str, Any]:
        """
        Analyze training history and learning curves.
        
        Returns:
            Dictionary with learning curve analysis
        """
        print("\n" + "=" * 60)
        print("ANALYZING LEARNING CURVES")
        print("=" * 60)
        
        history_path = self.results_dir / history_file
        if not history_path.exists():
            print(f"Warning: Training history file not found: {history_path}")
            return {}
        
        # Load training history
        history = pd.read_csv(history_path)
        
        analysis = {
            'total_epochs': len(history),
            'final_train_loss': float(history['loss'].iloc[-1]),
            'final_val_loss': float(history['val_loss'].iloc[-1]),
            'final_train_auc': float(history['auc'].iloc[-1]),
            'final_val_auc': float(history['val_auc'].iloc[-1]),
            'best_val_auc': float(history['val_auc'].max()),
            'best_epoch': int(history['val_auc'].idxmax()) + 1
        }
        
        # Check for overfitting
        train_val_gap = analysis['final_train_auc'] - analysis['final_val_auc']
        analysis['overfitting_gap'] = float(train_val_gap)
        
        if train_val_gap > 0.05:
            analysis['overfitting_status'] = "Significant overfitting detected"
        elif train_val_gap > 0.02:
            analysis['overfitting_status'] = "Mild overfitting detected"
        else:
            analysis['overfitting_status'] = "No significant overfitting"
        
        # Convergence analysis
        last_10_epochs = history['val_auc'].tail(10)
        auc_std = last_10_epochs.std()
        
        if auc_std < 0.001:
            analysis['convergence_status'] = "Fully converged"
        elif auc_std < 0.005:
            analysis['convergence_status'] = "Well converged"
        else:
            analysis['convergence_status'] = "Still improving"
        
        print(f"Training completed after {analysis['total_epochs']} epochs")
        print(f"Best validation AUC: {analysis['best_val_auc']:.4f} at epoch {analysis['best_epoch']}")
        print(f"Final train AUC: {analysis['final_train_auc']:.4f}")
        print(f"Final validation AUC: {analysis['final_val_auc']:.4f}")
        print(f"Train-validation gap: {analysis['overfitting_gap']:.4f}")
        print(f"Overfitting status: {analysis['overfitting_status']}")
        print(f"Convergence status: {analysis['convergence_status']}")
        
        # Save analysis
        self._save_analysis(analysis, "learning_curve_analysis.json")
        
        return analysis
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Returns:
            Complete performance analysis
        """
        print("\n" + "=" * 60)
        print("GENERATING PERFORMANCE SUMMARY")
        print("=" * 60)
        
        summary = {}
        
        # Load final results
        try:
            with open(self.results_dir / "final_tuned_results.json", 'r') as f:
                tuned_results = json.load(f)
            summary['tuned_model'] = tuned_results
        except FileNotFoundError:
            print("Warning: Final tuned results not found")
        
        try:
            with open(self.results_dir / "baseline_untuned_results.json", 'r') as f:
                baseline_results = json.load(f)
            summary['baseline_model'] = baseline_results
        except FileNotFoundError:
            print("Warning: Baseline results not found")
        
        # Load comparison results
        try:
            comparison_df = pd.read_csv(self.results_dir / "comparison_results.csv")
            summary['model_rankings'] = comparison_df.to_dict('records')
        except FileNotFoundError:
            print("Warning: Comparison results not found")
        
        # Calculate improvements if both models exist
        if 'tuned_model' in summary and 'baseline_model' in summary:
            tuned_auc = summary['tuned_model']['auc_roc']
            baseline_auc = summary['baseline_model']['auc_roc']
            
            summary['improvements'] = {
                'absolute_auc_improvement': tuned_auc - baseline_auc,
                'relative_auc_improvement': (tuned_auc - baseline_auc) / baseline_auc * 100,
                'tuning_effectiveness': 'Effective' if tuned_auc > baseline_auc else 'Limited'
            }
            
            print(f"Tuning Results:")
            print(f"  Baseline AUC: {baseline_auc:.4f}")
            print(f"  Tuned AUC: {tuned_auc:.4f}")
            print(f"  Absolute improvement: {summary['improvements']['absolute_auc_improvement']:+.4f}")
            print(f"  Relative improvement: {summary['improvements']['relative_auc_improvement']:+.2f}%")
        
        # Save complete summary
        self._save_analysis(summary, "performance_summary.json")
        
        return summary
    
    def _save_analysis(self, analysis: Dict[str, Any], filename: str):
        """Save analysis results to JSON file."""
        output_path = self.results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Analysis saved to {filename}")


# Utility functions for detailed analysis
def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Calculate detailed confusion matrix metrics."""
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }


def calculate_roc_curve_data(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, List[float]]:
    """Calculate ROC curve data for visualization."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    return {
        'false_positive_rate': fpr.tolist(),
        'true_positive_rate': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc_score': float(auc_score)
    }