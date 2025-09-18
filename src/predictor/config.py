"""
Configuration Management
========================

This module handles all configuration settings for the Smoke-Free Predictor application.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for machine learning model parameters."""
    # Random Forest parameters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    
    # Training parameters
    test_size: float = 0.2
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Feature engineering
    feature_scaling: bool = True
    handle_missing_values: bool = True
    encode_categorical: bool = True


@dataclass 
class DataConfig:
    """Configuration for data processing and paths."""
    # Data paths
    data_dir: str = "data"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    
    # File formats
    input_format: str = "csv"
    output_format: str = "csv"
    model_format: str = "joblib"
    
    # Data processing
    missing_value_strategy: str = "median"  # median, mean, mode, drop
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Feature columns (customize based on your data)
    target_column: str = "smoke_free_outcome"
    id_column: Optional[str] = "participant_id"
    
    # Expected feature categories
    numeric_features: List[str] = None
    categorical_features: List[str] = None
    datetime_features: List[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_output: bool = True
    log_dir: str = "logs"
    log_filename: str = "smoke_free_predictor.log"


class Config:
    """Main configuration class that combines all configuration settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file (JSON/YAML)
        """
        # Initialize sub-configs
        self.model = ModelConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        
        # Application settings
        self.app_name = "Smoke-Free Predictor"
        self.version = "1.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
            
        # Override with environment variables
        self.load_from_env()
        
        # Ensure directories exist
        self.create_directories()
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        # Implementation would depend on file format
        # For now, we'll skip this and rely on defaults + env vars
        pass
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Model configuration from environment
        self.model.n_estimators = int(os.getenv("MODEL_N_ESTIMATORS", self.model.n_estimators))
        self.model.max_depth = self._parse_optional_int(os.getenv("MODEL_MAX_DEPTH"))
        self.model.random_state = int(os.getenv("MODEL_RANDOM_STATE", self.model.random_state))
        
        # Data configuration from environment  
        self.data.data_dir = os.getenv("DATA_DIR", self.data.data_dir)
        self.data.models_dir = os.getenv("MODELS_DIR", self.data.models_dir)
        self.data.outputs_dir = os.getenv("OUTPUTS_DIR", self.data.outputs_dir)
        self.data.target_column = os.getenv("TARGET_COLUMN", self.data.target_column)
        
        # Logging configuration from environment
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.log_dir = os.getenv("LOG_DIR", self.logging.log_dir)
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.data_dir,
            self.data.models_dir, 
            self.data.outputs_dir,
            self.logging.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _parse_optional_int(self, value: Optional[str]) -> Optional[int]:
        """Parse optional integer from string."""
        if value is None or value.lower() in ('none', 'null', ''):
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'min_samples_leaf': self.model.min_samples_leaf,
                'random_state': self.model.random_state,
                'test_size': self.model.test_size,
                'validation_split': self.model.validation_split,
                'cross_validation_folds': self.model.cross_validation_folds,
                'feature_scaling': self.model.feature_scaling,
                'handle_missing_values': self.model.handle_missing_values,
                'encode_categorical': self.model.encode_categorical
            },
            'data': {
                'data_dir': self.data.data_dir,
                'models_dir': self.data.models_dir,
                'outputs_dir': self.data.outputs_dir,
                'input_format': self.data.input_format,
                'output_format': self.data.output_format,
                'model_format': self.data.model_format,
                'missing_value_strategy': self.data.missing_value_strategy,
                'outlier_detection': self.data.outlier_detection,
                'outlier_method': self.data.outlier_method,
                'target_column': self.data.target_column,
                'id_column': self.data.id_column
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_output': self.logging.file_output,
                'log_dir': self.logging.log_dir,
                'log_filename': self.logging.log_filename
            },
            'app_name': self.app_name,
            'version': self.version,
            'debug': self.debug
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(app_name='{self.app_name}', version='{self.version}', debug={self.debug})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config({self.to_dict()})"


# Global configuration instance
config = Config()