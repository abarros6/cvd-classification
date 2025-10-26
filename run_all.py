"""
Master script to run the entire pipeline.
Executes all steps: preprocessing, training, evaluation, visualization.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in {description}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")


def main():
    """Run the complete pipeline."""
    print("="*80)
    print("CARDIOVASCULAR DISEASE CLASSIFICATION - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis script will run the entire pipeline:")
    print("  1. Data Preprocessing")
    print("  2. Model Training (3 algorithms)")
    print("  3. Model Evaluation")
    print("  4. Visualization Generation")
    
    print("\nStarting pipeline execution...")
    
    # Check if data file exists
    if not os.path.exists('data/cardio_train.csv'):
        print("\n❌ Error: data/cardio_train.csv not found!")
        print("Please download the dataset and place it in the data/ folder.")
        print("\nDataset URL: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        sys.exit(1)
    
    # Step 1: Preprocessing
    run_command("python3 src/data_preprocessing.py", "Data Preprocessing")
    
    # Step 2: Training
    run_command("python3 src/model_training.py", "Model Training")
    
    # Step 3: Evaluation
    run_command("python3 src/evaluation.py", "Model Evaluation")
    
    # Step 4: Visualization
    run_command("python3 src/visualization.py", "Visualization Generation")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nResults located in:")
    print("  • results/model_results.csv - Performance metrics")
    print("  • figures/*.png - Visualization plots")
    print("  • models/*.pkl - Trained models")


if __name__ == "__main__":
    main()