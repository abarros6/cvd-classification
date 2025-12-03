"""
Hyperparameter Tuning Implementation

Implements the 3-phase hyperparameter tuning strategy for deep neural networks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import time
from pathlib import Path

from deep_nn_model import DeepNeuralNetworkBuilder, create_callbacks
from deep_nn_config import (
    HyperparameterConfig,
    generate_architecture_configs,
    generate_regularization_configs,
    generate_training_configs
)


class HyperparameterTuner:
    """Systematic hyperparameter tuning for deep neural networks."""
    
    def __init__(self, 
                 results_dir: str = "../results",
                 models_dir: str = "../models",
                 random_state: int = 42):
        """Initialize the hyperparameter tuner."""
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.random_state = random_state
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        (self.models_dir / "tuning_checkpoints").mkdir(exist_ok=True)
        
        # Initialize model builder
        self.builder = DeepNeuralNetworkBuilder(random_state=random_state)
        
        # Results storage
        self.tuning_results = []
    
    def phase1_architecture_search(self, 
                                 X_train: np.ndarray, 
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 verbose: int = 1) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Phase 1: Grid search over network architectures.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            verbose: Verbosity level
            
        Returns:
            Tuple of (best_config, results_dataframe)
        """
        print("=" * 60)
        print("PHASE 1: ARCHITECTURE SEARCH")
        print("=" * 60)
        
        configs = generate_architecture_configs()
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nExperiment {i+1}/{len(configs)}: "
                  f"{config['num_layers']} layers, {config['neurons_per_layer']} neurons")
            
            # Train model
            start_time = time.time()
            result = self._train_and_evaluate(
                config, X_train, y_train, X_val, y_val,
                model_name=f"phase1_arch_{i+1}",
                verbose=verbose
            )
            training_time = time.time() - start_time
            
            # Store results
            result.update({
                'phase': 1,
                'experiment': i + 1,
                'num_layers': config['num_layers'],
                'neurons_per_layer': config['neurons_per_layer'],
                'training_time': training_time
            })
            results.append(result)
            
            print(f"Validation AUC: {result['val_auc']:.4f}, "
                  f"Time: {training_time:.1f}s")
        
        # Convert to DataFrame and find best
        results_df = pd.DataFrame(results)
        best_idx = results_df['val_auc'].idxmax()
        best_config = configs[best_idx]
        
        # Save results
        results_df.to_csv(self.results_dir / "phase1_architecture_results.csv", index=False)
        
        print(f"\nBest architecture: {best_config['num_layers']} layers, "
              f"{best_config['neurons_per_layer']} neurons")
        print(f"Best validation AUC: {results_df.loc[best_idx, 'val_auc']:.4f}")
        
        return best_config, results_df
    
    def phase2_regularization_search(self,
                                   best_arch_config: Dict[str, Any],
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_val: np.ndarray,
                                   y_val: np.ndarray,
                                   verbose: int = 1) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Phase 2: Grid search over regularization parameters.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: REGULARIZATION SEARCH")
        print("=" * 60)
        
        configs = generate_regularization_configs(best_arch_config)
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nExperiment {i+1}/{len(configs)}: "
                  f"dropout={config['dropout_rate']}, l2={config['l2_regularization']}")
            
            # Train model
            start_time = time.time()
            result = self._train_and_evaluate(
                config, X_train, y_train, X_val, y_val,
                model_name=f"phase2_reg_{i+1}",
                verbose=verbose
            )
            training_time = time.time() - start_time
            
            # Store results
            result.update({
                'phase': 2,
                'experiment': i + 1,
                'dropout_rate': config['dropout_rate'],
                'l2_regularization': config['l2_regularization'],
                'training_time': training_time
            })
            results.append(result)
            
            print(f"Validation AUC: {result['val_auc']:.4f}, "
                  f"Time: {training_time:.1f}s")
        
        # Convert to DataFrame and find best
        results_df = pd.DataFrame(results)
        best_idx = results_df['val_auc'].idxmax()
        best_config = configs[best_idx]
        
        # Save results
        results_df.to_csv(self.results_dir / "phase2_regularization_results.csv", index=False)
        
        print(f"\nBest regularization: dropout={best_config['dropout_rate']}, "
              f"l2={best_config['l2_regularization']}")
        print(f"Best validation AUC: {results_df.loc[best_idx, 'val_auc']:.4f}")
        
        return best_config, results_df
    
    def phase3_training_optimization(self,
                                   best_reg_config: Dict[str, Any],
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_val: np.ndarray,
                                   y_val: np.ndarray,
                                   n_trials: int = 20,
                                   verbose: int = 1) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Phase 3: Random search over training optimization parameters.
        """
        print("\n" + "=" * 60)
        print("PHASE 3: TRAINING OPTIMIZATION")
        print("=" * 60)
        
        configs = generate_training_configs(best_reg_config, n_trials)
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nTrial {i+1}/{len(configs)}: "
                  f"lr={config['learning_rate']:.4f}, "
                  f"batch={config['batch_size']}, "
                  f"opt={config['optimizer']}")
            
            # Train model
            start_time = time.time()
            result = self._train_and_evaluate(
                config, X_train, y_train, X_val, y_val,
                model_name=f"phase3_train_{i+1}",
                verbose=verbose
            )
            training_time = time.time() - start_time
            
            # Store results
            result.update({
                'phase': 3,
                'trial': config['trial'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'optimizer': config['optimizer'],
                'training_time': training_time
            })
            results.append(result)
            
            print(f"Validation AUC: {result['val_auc']:.4f}, "
                  f"Time: {training_time:.1f}s")
        
        # Convert to DataFrame and find best
        results_df = pd.DataFrame(results)
        best_idx = results_df['val_auc'].idxmax()
        best_config = configs[best_idx]
        
        # Save results
        results_df.to_csv(self.results_dir / "phase3_training_results.csv", index=False)
        
        print(f"\nBest training config: lr={best_config['learning_rate']:.4f}, "
              f"batch={best_config['batch_size']}, opt={best_config['optimizer']}")
        print(f"Best validation AUC: {results_df.loc[best_idx, 'val_auc']:.4f}")
        
        return best_config, results_df
    
    def _train_and_evaluate(self,
                          config: Dict[str, Any],
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          model_name: str,
                          verbose: int = 0) -> Dict[str, float]:
        """
        Train and evaluate a single model configuration.
        
        Returns:
            Dictionary with validation metrics
        """
        # Build model
        model = self.builder.build_model(config)
        
        # Create callbacks
        model_path = self.models_dir / "tuning_checkpoints" / f"{model_name}.h5"
        callbacks = create_callbacks(config, str(model_path))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Get best metrics
        best_epoch = np.argmax(history.history['val_auc'])
        
        return {
            'val_loss': history.history['val_loss'][best_epoch],
            'val_accuracy': history.history['val_accuracy'][best_epoch],
            'val_auc': history.history['val_auc'][best_epoch],
            'train_loss': history.history['loss'][best_epoch],
            'train_accuracy': history.history['accuracy'][best_epoch],
            'train_auc': history.history['auc'][best_epoch],
            'best_epoch': best_epoch + 1,
            'total_epochs': len(history.history['loss'])
        }
    
    def run_full_tuning(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray, 
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       verbose: int = 1) -> Dict[str, Any]:
        """
        Run complete 3-phase hyperparameter tuning.
        
        Returns:
            Best configuration from all phases
        """
        print("Starting comprehensive hyperparameter tuning...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Phase 1: Architecture search
        best_arch_config, arch_results = self.phase1_architecture_search(
            X_train, y_train, X_val, y_val, verbose
        )
        
        # Phase 2: Regularization search
        best_reg_config, reg_results = self.phase2_regularization_search(
            best_arch_config, X_train, y_train, X_val, y_val, verbose
        )
        
        # Phase 3: Training optimization
        best_final_config, train_results = self.phase3_training_optimization(
            best_reg_config, X_train, y_train, X_val, y_val, verbose=verbose
        )
        
        # Combine all results
        all_results = pd.concat([arch_results, reg_results, train_results], 
                               ignore_index=True)
        all_results.to_csv(self.results_dir / "complete_tuning_results.csv", index=False)
        
        # Save best configuration
        import json
        with open(self.results_dir / "best_model_config.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_config = {}
            for k, v in best_final_config.items():
                if isinstance(v, np.ndarray):
                    json_config[k] = v.tolist()
                elif isinstance(v, np.integer):
                    json_config[k] = int(v)
                elif isinstance(v, np.floating):
                    json_config[k] = float(v)
                else:
                    json_config[k] = v
            json.dump(json_config, f, indent=2)
        
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING COMPLETE!")
        print("=" * 60)
        print(f"Best overall validation AUC: {all_results['val_auc'].max():.4f}")
        print(f"Results saved to: {self.results_dir}")
        
        return best_final_config