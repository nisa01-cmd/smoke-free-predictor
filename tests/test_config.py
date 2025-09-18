"""
Unit tests for the configuration module.
"""

import os
import pytest
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config, ModelConfig, DataConfig, LoggingConfig


class TestModelConfig:
    """Test the ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()
        
        assert config.n_estimators == 100
        assert config.max_depth is None
        assert config.min_samples_split == 2
        assert config.min_samples_leaf == 1
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.validation_split == 0.2
        assert config.cross_validation_folds == 5
        assert config.feature_scaling is True
        assert config.handle_missing_values is True
        assert config.encode_categorical is True


class TestDataConfig:
    """Test the DataConfig dataclass."""
    
    def test_default_values(self):
        """Test that DataConfig has correct default values."""
        config = DataConfig()
        
        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.outputs_dir == "outputs"
        assert config.input_format == "csv"
        assert config.output_format == "csv"
        assert config.model_format == "joblib"
        assert config.missing_value_strategy == "median"
        assert config.outlier_detection is True
        assert config.outlier_method == "iqr"
        assert config.target_column == "smoke_free_outcome"
        assert config.id_column == "participant_id"


class TestLoggingConfig:
    """Test the LoggingConfig dataclass."""
    
    def test_default_values(self):
        """Test that LoggingConfig has correct default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file_output is True
        assert config.log_dir == "logs"
        assert config.log_filename == "smoke_free_predictor.log"


class TestConfig:
    """Test the main Config class."""
    
    def test_default_initialization(self):
        """Test that Config initializes with correct defaults."""
        config = Config()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.app_name == "Smoke-Free Predictor"
        assert config.version == "1.0.0"
        assert config.debug is False
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        # Set test environment variables
        os.environ["MODEL_N_ESTIMATORS"] = "200"
        os.environ["MODEL_RANDOM_STATE"] = "123"
        os.environ["DATA_DIR"] = "test_data"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        try:
            config = Config()
            
            assert config.model.n_estimators == 200
            assert config.model.random_state == 123
            assert config.data.data_dir == "test_data"
            assert config.logging.level == "DEBUG"
            
        finally:
            # Clean up environment variables
            for var in ["MODEL_N_ESTIMATORS", "MODEL_RANDOM_STATE", "DATA_DIR", "LOG_LEVEL"]:
                os.environ.pop(var, None)
    
    def test_debug_environment_variable(self):
        """Test that DEBUG environment variable works."""
        os.environ["DEBUG"] = "true"
        
        try:
            config = Config()
            assert config.debug is True
            
        finally:
            os.environ.pop("DEBUG", None)
    
    def test_optional_int_parsing(self):
        """Test the _parse_optional_int method."""
        config = Config()
        
        assert config._parse_optional_int("10") == 10
        assert config._parse_optional_int("none") is None
        assert config._parse_optional_int("null") is None
        assert config._parse_optional_int("") is None
        assert config._parse_optional_int(None) is None
        assert config._parse_optional_int("invalid") is None
    
    def test_to_dict(self):
        """Test that to_dict returns the correct structure."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "data" in config_dict
        assert "logging" in config_dict
        assert "app_name" in config_dict
        assert "version" in config_dict
        assert "debug" in config_dict
        
        # Check model section
        model_config = config_dict["model"]
        assert "n_estimators" in model_config
        assert "random_state" in model_config
        assert model_config["n_estimators"] == 100
    
    def test_string_representation(self):
        """Test string representations."""
        config = Config()
        
        str_repr = str(config)
        assert "Config" in str_repr
        assert "Smoke-Free Predictor" in str_repr
        assert "1.0.0" in str_repr
        
        repr_str = repr(config)
        assert "Config" in repr_str
    
    def test_directory_creation(self):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override directories to use temp directory
            os.environ["DATA_DIR"] = f"{temp_dir}/data"
            os.environ["MODELS_DIR"] = f"{temp_dir}/models"
            os.environ["OUTPUTS_DIR"] = f"{temp_dir}/outputs"
            os.environ["LOG_DIR"] = f"{temp_dir}/logs"
            
            try:
                config = Config()
                
                # Check that directories were created
                assert Path(f"{temp_dir}/data").exists()
                assert Path(f"{temp_dir}/models").exists()
                assert Path(f"{temp_dir}/outputs").exists()
                assert Path(f"{temp_dir}/logs").exists()
                
            finally:
                for var in ["DATA_DIR", "MODELS_DIR", "OUTPUTS_DIR", "LOG_DIR"]:
                    os.environ.pop(var, None)


if __name__ == "__main__":
    pytest.main([__file__])