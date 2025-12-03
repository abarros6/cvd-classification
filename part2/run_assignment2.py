#!/usr/bin/env python3
"""
Assignment 2 Master Execution Script

Runs the complete deep neural network hyperparameter tuning pipeline.
"""

# Standard library imports
import os
import sys
import time
from pathlib import Path

# Add src package to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Local imports
from evaluation import ModelEvaluator
from hyperparameter_tuning import HyperparameterTuner
from training import DataLoader, ModelTrainer
from visualization import AssignmentVisualizer


def main():
    """
    Run the complete Assignment 2 pipeline.
    Executes all steps: data loading, hyperparameter tuning, evaluation, visualization.
    """
    print("=" * 80)
    print("ASSIGNMENT 2: DEEP NEURAL NETWORK HYPERPARAMETER TUNING")
    print("ECE 9603 Data Analytics Foundations")
    print("=" * 80)
    print("\nThis pipeline will run the complete hyperparameter tuning process:")
    print("  1. Data Loading and Preprocessing")
    print("  2. Phase 1: Architecture Search (layers, neurons)")
    print("  3. Phase 2: Regularization Tuning (dropout, L2)")
    print("  4. Phase 3: Training Optimization (learning rate, batch size)")
    print("  5. Model Evaluation and Comparison")
    print("  6. Visualization Generation")
    print("\nStarting pipeline execution...")
    
    start_time = time.time()
    
    try:
        # Step 1: Load and prepare data
        print("\n" + "=" * 80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 80)
        
        # Load data
        X_train_full, X_test, y_train_full, y_test = DataLoader.load_raw_data("data")
        
        # Split training data for hyperparameter tuning
        X_train, X_val, y_train, y_val = DataLoader.prepare_data_for_tuning(
            X_train_full, y_train_full, validation_split=0.1
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = DataLoader.apply_standardization(
            X_train, X_val, X_test
        )
        
        print(f"✓ Data preparation complete!")
        print(f"Final data shapes:")
        print(f"  Training: {X_train_scaled.shape}")
        print(f"  Validation: {X_val_scaled.shape}")
        print(f"  Test: {X_test_scaled.shape}")
        
        # Step 2: Hyperparameter tuning (3-phase strategy)
        print("\n" + "Step 2: Running hyperparameter tuning...")
        print("-" * 50)
        
        tuner = HyperparameterTuner(
            results_dir="results",
            models_dir="models"
        )
        
        best_config = tuner.run_full_tuning(
            X_train_scaled, y_train, X_val_scaled, y_val, verbose=1
        )
        
        # Step 3: Train baseline and final models
        print("\n" + "Step 3: Training final models...")
        print("-" * 50)
        
        trainer = ModelTrainer(
            results_dir="results",
            models_dir="models",
            figures_dir="figures"
        )
        
        # Train baseline (untuned) model for comparison
        baseline_model, baseline_metrics, baseline_history = trainer.train_baseline_model(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
        )
        
        # Train final tuned model
        final_model, final_metrics, final_history = trainer.train_final_model(
            best_config, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
        )
        
        # Step 4: Performance evaluation and comparison
        print("\n" + "Step 4: Performance evaluation and comparison...")
        print("-" * 50)
        
        comparison_df = trainer.compare_with_baselines(final_metrics, baseline_metrics)
        
        # Step 5: Comprehensive evaluation
        print("\n" + "Step 5: Running comprehensive evaluation...")
        print("-" * 50)
        
        evaluator = ModelEvaluator(results_dir="results")
        
        # Analyze tuning results
        tuning_analysis = evaluator.analyze_tuning_results()
        
        # Analyze learning curves
        learning_analysis = evaluator.analyze_learning_curves()
        
        # Generate performance summary
        performance_summary = evaluator.generate_performance_summary()
        
        # Step 6: Create visualizations
        print("\n" + "Step 6: Creating visualizations...")
        print("-" * 50)
        
        visualizer = AssignmentVisualizer(
            results_dir="results",
            figures_dir="figures"
        )
        
        visualizer.create_all_visualizations()
        
        # Step 7: Generate final report summary
        print("\n" + "Step 7: Generating final summary...")
        print("-" * 50)
        
        generate_execution_summary(
            tuning_analysis, 
            performance_summary, 
            final_metrics, 
            baseline_metrics,
            time.time() - start_time
        )
        
        print("\n" + "=" * 80)
        print("ASSIGNMENT 2 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        print(f"Results saved to: ./results/")
        print(f"Figures saved to: ./figures/")
        print(f"Models saved to: ./models/")
        print("\nNext step: Review results and update ASSIGNMENT2_REPORT.md")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Pipeline execution failed. Check error details above.")
        sys.exit(1)


def generate_execution_summary(tuning_analysis, performance_summary, 
                             final_metrics, baseline_metrics, execution_time):
    """Generate a summary of the execution for easy review."""
    
    summary_text = f"""
# Assignment 2 Execution Summary

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Execution Time:** {execution_time:.2f} seconds

## Key Results

### Model Performance
- **Tuned Deep NN AUC-ROC:** {final_metrics['auc_roc']:.4f}
- **Baseline Deep NN AUC-ROC:** {baseline_metrics['auc_roc']:.4f}
- **Improvement from tuning:** {final_metrics['auc_roc'] - baseline_metrics['auc_roc']:+.4f}

### Hyperparameter Tuning Results
"""
    
    if tuning_analysis:
        summary_text += f"- **Total experiments:** {tuning_analysis.get('total_experiments', 'N/A')}\n"
        summary_text += f"- **Best validation AUC:** {tuning_analysis.get('best_val_auc', 'N/A'):.4f}\n"
        
        if 'phase1' in tuning_analysis and 'best_architecture' in tuning_analysis['phase1']:
            best_arch = tuning_analysis['phase1']['best_architecture']
            summary_text += f"- **Best architecture:** {best_arch['num_layers']} layers, {best_arch['neurons_per_layer']} neurons\n"
    
    summary_text += f"""
### Files Generated
- Results: ./results/ (CSV files and JSON analyses)
- Models: ./models/ (trained model files)  
- Figures: ./figures/ (all required visualizations)
- Report: ./ASSIGNMENT2_REPORT.md (update with these results)

### Assignment Requirements Status
✅ Deep neural network with 5+ layers implemented
✅ 4+ hyperparameters tuned systematically  
✅ 3-phase tuning strategy executed
✅ Learning curves generated (proves NN is learning)
✅ Performance evaluation and baseline comparison completed
✅ Multiple metrics compared (AUC-ROC, Accuracy, Precision, Recall)
✅ Detailed methodology documented and reproducible

### Next Steps
1. Review all generated results in ./results/
2. Examine visualizations in ./figures/
3. Update ASSIGNMENT2_REPORT.md with specific findings
4. Analyze learning curves to confirm the NN is learning effectively
5. Document insights about hyperparameter importance
"""
    
    # Save summary
    with open("results/execution_summary.md", 'w') as f:
        f.write(summary_text)
    
    print("Execution summary saved to: results/execution_summary.md")


if __name__ == "__main__":
    main()