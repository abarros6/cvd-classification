"""
Deep Neural Network Model Builder

Implements flexible deep neural network architectures for cardiovascular disease classification.
"""

# Standard library imports
from typing import List, Dict, Any, Tuple

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers


class DeepNeuralNetworkBuilder:
    """Builder class for creating deep neural network models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the neural network builder."""
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def build_model(self, config: Dict[str, Any]) -> keras.Model:
        """Build and compile the neural network model."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(config['input_dim'],)))
        
        # Hidden layers
        layer_sizes = config.get('layer_sizes', 
                                [config['neurons_per_layer']] * config['num_layers'])
        
        for i, neurons in enumerate(layer_sizes):
            # Dense layer with L2 regularization
            model.add(layers.Dense(
                neurons,
                activation=config.get('activation', 'relu'),
                kernel_regularizer=regularizers.l2(config.get('l2_regularization', 0.0)),
                name=f'hidden_{i+1}'
            ))
            
            # Batch normalization (optional)
            if config.get('use_batch_norm', False):
                model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Dropout for regularization
            dropout_rate = config.get('dropout_rate', 0.0)
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer for binary classification
        model.add(layers.Dense(
            1, 
            activation=config.get('output_activation', 'sigmoid'),
            name='output'
        ))
        
        # Compile model
        optimizer = self._get_optimizer(config)
        
        model.compile(
            optimizer=optimizer,
            loss=config.get('loss', 'binary_crossentropy'),
            metrics=self._get_metrics(config.get('metrics', ['accuracy', 'auc']))
        )
        
        return model
    
    def _get_optimizer(self, config: Dict[str, Any]) -> keras.optimizers.Optimizer:
        """Get optimizer based on configuration."""
        optimizer_name = config.get('optimizer', 'adam')
        learning_rate = config.get('learning_rate', 0.001)
        
        if optimizer_name.lower() == 'adam':
            return optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            return optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_metrics(self, metric_names: List[str]) -> List[keras.metrics.Metric]:
        """Convert metric names to Keras metric objects."""
        metrics = []
        for name in metric_names:
            if name.lower() == 'accuracy':
                metrics.append('accuracy')
            elif name.lower() == 'auc':
                metrics.append(keras.metrics.AUC(name='auc'))
            elif name.lower() == 'precision':
                metrics.append(keras.metrics.Precision(name='precision'))
            elif name.lower() == 'recall':
                metrics.append(keras.metrics.Recall(name='recall'))
            else:
                metrics.append(name)  # Pass through unknown metrics
        
        return metrics
    
    def get_model_summary(self, model: keras.Model) -> str:
        """Get detailed model summary as string."""
        import io
        import contextlib
        
        # Capture model summary
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            model.summary()
        summary_string = f.getvalue()
        
        return summary_string
    
    def count_parameters(self, model: keras.Model) -> Tuple[int, int]:
        """
        Count trainable and total parameters in model.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_count = int(
            np.sum([keras.backend.count_params(p) for p in model.trainable_weights])
        )
        non_trainable_count = int(
            np.sum([keras.backend.count_params(p) for p in model.non_trainable_weights])
        )
        
        return trainable_count, trainable_count + non_trainable_count


def create_callbacks(config: Dict[str, Any], model_save_path: str = None) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks based on configuration.
    
    Args:
        config: Configuration dictionary
        model_save_path: Path to save best model (optional)
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.get('early_stopping_monitor', 'val_loss'),
        patience=config.get('early_stopping_patience', 10),
        restore_best_weights=True,
        mode=config.get('early_stopping_mode', 'min'),
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpointing (if path provided)
    if model_save_path:
        checkpoint = keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor=config.get('early_stopping_monitor', 'val_loss'),
            save_best_only=True,
            mode=config.get('early_stopping_mode', 'min'),
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Learning rate reduction on plateau (optional)
    if config.get('reduce_lr_on_plateau', False):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=config.get('early_stopping_monitor', 'val_loss'),
            factor=0.2,
            patience=config.get('early_stopping_patience', 10) // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    return callbacks


# Example usage and testing
if __name__ == "__main__":
    # Test the model builder
    from deep_nn_config import HyperparameterConfig
    
    # Create builder
    builder = DeepNeuralNetworkBuilder()
    
    # Test with baseline config
    baseline_config = HyperparameterConfig.get_baseline_config()
    baseline_model = builder.build_model(baseline_config)
    
    print("Baseline Model Summary:")
    print(builder.get_model_summary(baseline_model))
    
    trainable, total = builder.count_parameters(baseline_model)
    print(f"Parameters - Trainable: {trainable:,}, Total: {total:,}")
    
    # Test with deep config
    deep_config = baseline_config.copy()
    deep_config.update({
        'num_layers': 7,
        'neurons_per_layer': 128,
        'layer_sizes': [128] * 7,
        'dropout_rate': 0.3,
        'l2_regularization': 0.001
    })
    
    deep_model = builder.build_model(deep_config)
    
    print("\nDeep Model Summary:")
    print(builder.get_model_summary(deep_model))
    
    trainable, total = builder.count_parameters(deep_model)
    print(f"Parameters - Trainable: {trainable:,}, Total: {total:,}")