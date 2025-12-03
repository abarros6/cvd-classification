"""
Deep Neural Network Configuration

Defines hyperparameter configurations and search spaces for Assignment 2.
"""

# Standard library imports
from typing import Dict, List, Any

# Third-party imports
import numpy as np


class HyperparameterConfig:
    """Configuration class for deep neural network hyperparameters."""
    
    # Architecture search space (Phase 1)
    ARCHITECTURE_SEARCH = {
        'num_layers': [3, 5, 7],  # Reduced from 4 to 3 options
        'neurons_per_layer': [64, 128],  # Reduced from 3 to 2 options
        'layer_pattern': 'constant'  # constant, decreasing, pyramid
    }
    
    # Regularization search space (Phase 2)
    REGULARIZATION_SEARCH = {
        'dropout_rate': [0.0, 0.3],  # Reduced to 2 options
        'l2_regularization': [0.0, 0.001]  # Reduced to 2 options
    }
    
    # Training optimization search space (Phase 3)
    TRAINING_SEARCH = {
        'learning_rate': {
            'type': 'log_uniform',
            'low': 1e-4,
            'high': 1e-1
        },
        'batch_size': [32, 64, 128, 256],
        'optimizer': ['adam', 'rmsprop']
    }
    
    # Fixed parameters
    FIXED_PARAMS = {
        'input_dim': 12,  # 11 original features + BMI
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy', 'auc'],
        'epochs': 10,  # Further reduced for faster execution
        'early_stopping_patience': 3,  # Further reduced patience
        'early_stopping_monitor': 'val_auc',
        'early_stopping_mode': 'max',
        'validation_split': 0.1,  # 10% of training data for validation
        'random_state': 42
    }
    
    @classmethod
    def get_baseline_config(cls) -> Dict[str, Any]:
        """Get baseline neural network config."""
        return {
            'num_layers': 2,
            'neurons_per_layer': 64,
            'layer_sizes': [64, 32],  # Assignment 1 architecture
            'dropout_rate': 0.0,
            'l2_regularization': 0.0,
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            **cls.FIXED_PARAMS
        }
    
    @classmethod
    def calculate_layer_sizes(cls, 
                            num_layers: int, 
                            base_neurons: int, 
                            pattern: str = 'constant') -> List[int]:
        """Calculate layer sizes for the network."""
        if pattern == 'constant':
            return [base_neurons] * num_layers
        
        elif pattern == 'decreasing':
            # Gradually decrease from base_neurons to base_neurons//4
            sizes = []
            for i in range(num_layers):
                factor = (1 - i / (num_layers - 1)) * 0.75 + 0.25  # Scale from 1.0 to 0.25
                sizes.append(max(16, int(base_neurons * factor)))
            return sizes
        
        elif pattern == 'pyramid':
            # Increase to middle, then decrease
            sizes = []
            mid = num_layers // 2
            for i in range(num_layers):
                if i <= mid:
                    # Increasing phase
                    factor = 1 + (i / mid) * 0.5  # Scale from 1.0 to 1.5
                else:
                    # Decreasing phase
                    factor = 1.5 - ((i - mid) / (num_layers - mid - 1)) * 0.5  # Scale from 1.5 to 1.0
                sizes.append(max(16, int(base_neurons * factor)))
            return sizes
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    @classmethod
    def generate_random_training_config(cls, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate random configuration for Phase 3 training optimization.
        
        Parameters:
        -----------
        base_config : Dict[str, Any]
            Base configuration to extend
            
        Returns:
        --------
        Dict[str, Any]
            Configuration with randomized training parameters
        """
        config = base_config.copy()
        
        # Sample learning rate log-uniformly
        log_lr = np.random.uniform(
            np.log10(cls.TRAINING_SEARCH['learning_rate']['low']),
            np.log10(cls.TRAINING_SEARCH['learning_rate']['high'])
        )
        config['learning_rate'] = 10 ** log_lr
        
        # Sample other parameters uniformly
        config['batch_size'] = np.random.choice(cls.TRAINING_SEARCH['batch_size'])
        config['optimizer'] = np.random.choice(cls.TRAINING_SEARCH['optimizer'])
        
        return config


# Phase 1: Architecture configurations
def generate_architecture_configs():
    """
    Generate all architecture configurations for grid search.
    
    Returns:
    --------
    List[Dict[str, Any]]
        List of architecture configurations for Phase 1 tuning
    """
    configs = []
    
    for num_layers in HyperparameterConfig.ARCHITECTURE_SEARCH['num_layers']:
        for neurons in HyperparameterConfig.ARCHITECTURE_SEARCH['neurons_per_layer']:
            config = HyperparameterConfig.get_baseline_config()
            config.update({
                'num_layers': num_layers,
                'neurons_per_layer': neurons,
                'layer_sizes': HyperparameterConfig.calculate_layer_sizes(
                    num_layers, neurons, 'constant'
                ),
                'dropout_rate': 0.3,  # Fixed for Phase 1
                'l2_regularization': 0.001,  # Fixed for Phase 1
            })
            configs.append(config)
    
    return configs


# Phase 2: Regularization configurations
def generate_regularization_configs(best_arch_config):
    """
    Generate regularization configurations for grid search.
    
    Parameters:
    -----------
    best_arch_config : Dict[str, Any]
        Best architecture configuration from Phase 1
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of regularization configurations for Phase 2 tuning
    """
    configs = []
    
    for dropout in HyperparameterConfig.REGULARIZATION_SEARCH['dropout_rate']:
        for l2_reg in HyperparameterConfig.REGULARIZATION_SEARCH['l2_regularization']:
            config = best_arch_config.copy()
            config.update({
                'dropout_rate': dropout,
                'l2_regularization': l2_reg
            })
            configs.append(config)
    
    return configs


# Phase 3: Training optimization configurations
def generate_training_configs(best_reg_config, n_trials=3):
    """
    Generate random training configurations.
    
    Parameters:
    -----------
    best_reg_config : Dict[str, Any]
        Best regularization configuration from Phase 2
    n_trials : int, optional
        Number of random configurations to generate
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of training configurations for Phase 3 tuning
    """
    configs = []
    
    for trial in range(n_trials):
        config = HyperparameterConfig.generate_random_training_config(best_reg_config)
        config['trial'] = trial
        configs.append(config)
    
    return configs