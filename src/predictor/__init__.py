"""
Smoke-Free Predictor Interface
=============================

This module provides a high-level interface for making predictions using the
trained smoke-free prediction model. It combines the model and data processing
components to provide an easy-to-use prediction API.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from .model import SmokeFreeModel
from .data_processor import DataProcessor
from .config import config

logger = logging.getLogger(__name__)


class SmokeFreePredictor:
    """
    High-level prediction interface for the smoke-free predictor.
    
    This class provides methods to make predictions on new data using a 
    trained model and data preprocessing pipeline.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to a trained model file. If None, creates a new untrained model.
        """
        self.model = SmokeFreeModel()
        self.data_processor = DataProcessor()
        self.is_ready = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model file
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        logger.info(f"Loading model from: {model_path}")
        
        # Load the model
        self.model.load(model_path)
        
        # Try to load associated preprocessor
        preprocessor_path = Path(model_path).with_suffix('.preprocessor.joblib')
        if preprocessor_path.exists():
            self.data_processor.load_preprocessor(str(preprocessor_path))
            logger.info(f"Loaded preprocessor from: {preprocessor_path}")
        else:
            logger.warning("No preprocessor found. Data preprocessing may not work correctly.")
        
        self.is_ready = True
        logger.info("Model loaded successfully and predictor is ready!")
    
    def train(self, training_data_path: str, target_column: str = None, 
              validation_split: float = None, save_model: bool = True,
              model_output_path: str = None) -> Dict[str, Any]:
        """
        Train the predictor on provided data.
        
        Args:
            training_data_path: Path to training data file
            target_column: Name of target column (uses config default if None)
            validation_split: Fraction of data for validation
            save_model: Whether to save the trained model
            model_output_path: Where to save the model (uses default if None)
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Training predictor with data from: {training_data_path}")
        
        # Load and preprocess training data
        X, y = self.data_processor.load_and_preprocess(
            training_data_path, target_column
        )
        
        # Train the model
        training_results = self.model.train(X, y, validation_split)
        
        # Mark as ready
        self.is_ready = True
        
        # Save model if requested
        if save_model:
            output_path = model_output_path or config.data.models_dir + "/smoke_free_model.joblib"
            self.save_model(output_path)
        
        logger.info("Training completed successfully!")
        return training_results
    
    def predict(self, X: np.ndarray, include_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions on preprocessed feature data.
        
        Args:
            X: Preprocessed feature matrix
            include_probabilities: Whether to include prediction probabilities (classification only)
            
        Returns:
            Dictionary containing predictions and optionally probabilities
            
        Raises:
            ValueError: If predictor is not ready (no trained model)
        """
        if not self.is_ready:
            raise ValueError("Predictor not ready. Load a model or train first.")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        result = {
            'predictions': predictions.tolist(),
            'model_type': self.model.model_type,
            'prediction_count': len(predictions)
        }
        
        # Add probabilities for classification models if requested
        if (include_probabilities and 
            self.model.model_type == 'classification' and 
            hasattr(self.model.model, 'predict_proba')):
            
            probabilities = self.model.predict_proba(X)
            result['probabilities'] = probabilities.tolist()
            result['classes'] = self.model.target_classes.tolist() if self.model.target_classes is not None else None
        
        return result
    
    def predict_from_data(self, data: pd.DataFrame, include_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions on raw DataFrame.
        
        Args:
            data: Raw data DataFrame to make predictions on
            include_probabilities: Whether to include prediction probabilities
            
        Returns:
            Dictionary containing predictions and metadata
            
        Raises:
            ValueError: If predictor is not ready
        """
        if not self.is_ready:
            raise ValueError("Predictor not ready. Load a model or train first.")
        
        logger.info(f"Making predictions on {len(data)} samples...")
        
        # Preprocess the data
        X_processed = self.data_processor.transform_new_data(data)
        
        # Make predictions
        result = self.predict(X_processed, include_probabilities)
        
        # Add input data metadata
        result['input_shape'] = data.shape
        result['input_columns'] = data.columns.tolist()
        
        return result
    
    def predict_from_file(self, file_path: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions on data from a file.
        
        Args:
            file_path: Path to the data file
            include_probabilities: Whether to include prediction probabilities
            
        Returns:
            Dictionary containing predictions and metadata
        """
        logger.info(f"Making predictions from file: {file_path}")
        
        # Load the data
        data = self.data_processor.load_data(file_path)
        
        # Make predictions
        result = self.predict_from_data(data, include_probabilities)
        result['input_file'] = str(file_path)
        
        return result
    
    def predict_single(self, sample_data: Dict[str, Any], include_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make prediction on a single sample.
        
        Args:
            sample_data: Dictionary containing feature values for a single sample
            include_probabilities: Whether to include prediction probabilities
            
        Returns:
            Dictionary containing prediction result
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample_data])
        
        # Make prediction
        result = self.predict_from_data(df, include_probabilities)
        
        # Extract single prediction
        single_result = {
            'prediction': result['predictions'][0] if result['predictions'] else None,
            'model_type': result['model_type'],
            'input_features': sample_data
        }
        
        if 'probabilities' in result and result['probabilities']:
            single_result['probabilities'] = result['probabilities'][0]
            single_result['classes'] = result['classes']
        
        return single_result
    
    def evaluate_from_file(self, test_data_path: str) -> Dict[str, Any]:
        """
        Evaluate the model on test data from a file.
        
        Args:
            test_data_path: Path to test data file (must include target column)
            
        Returns:
            Dictionary containing evaluation metrics and results
            
        Raises:
            ValueError: If predictor is not ready or target column not found
        """
        if not self.is_ready:
            raise ValueError("Predictor not ready. Load a model or train first.")
        
        logger.info(f"Evaluating model on test data from: {test_data_path}")
        
        # Load and preprocess test data
        X_test, y_test = self.data_processor.load_and_preprocess(test_data_path)
        
        # Evaluate the model
        evaluation_results = self.model.evaluate(X_test, y_test)
        evaluation_results['test_data_path'] = str(test_data_path)
        evaluation_results['test_samples'] = len(y_test)
        
        return evaluation_results
    
    def save_model(self, model_path: str, include_preprocessor: bool = True) -> None:
        """
        Save the trained model and optionally the preprocessor.
        
        Args:
            model_path: Path to save the model
            include_preprocessor: Whether to save the data preprocessor
            
        Raises:
            ValueError: If no trained model to save
        """
        if not self.model.is_trained:
            raise ValueError("No trained model to save")
        
        logger.info(f"Saving model to: {model_path}")
        
        # Save the model
        self.model.save(model_path)
        
        # Save the preprocessor if requested and available
        if include_preprocessor and self.data_processor.preprocessor is not None:
            preprocessor_path = Path(model_path).with_suffix('.preprocessor.joblib')
            self.data_processor.save_preprocessor(str(preprocessor_path))
            logger.info(f"Saved preprocessor to: {preprocessor_path}")
    
    def save_predictions(self, predictions_result: Dict[str, Any], output_path: str,
                        include_input_data: bool = False, input_data: pd.DataFrame = None) -> None:
        """
        Save predictions to a file.
        
        Args:
            predictions_result: Result from predict_* methods
            output_path: Path to save predictions
            include_input_data: Whether to include original input data
            input_data: Original input data (required if include_input_data is True)
        """
        logger.info(f"Saving predictions to: {output_path}")
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'prediction': predictions_result['predictions']
        })
        
        # Add probabilities if available
        if 'probabilities' in predictions_result:
            probabilities = predictions_result['probabilities']
            classes = predictions_result.get('classes', [])
            
            if classes:
                # Add probability columns for each class
                for i, class_name in enumerate(classes):
                    predictions_df[f'probability_{class_name}'] = [prob[i] for prob in probabilities]
            else:
                # Add generic probability columns
                prob_array = np.array(probabilities)
                for i in range(prob_array.shape[1]):
                    predictions_df[f'probability_{i}'] = prob_array[:, i]
        
        # Add input data if requested and available
        if include_input_data and input_data is not None:
            if len(input_data) == len(predictions_df):
                predictions_df = pd.concat([input_data.reset_index(drop=True), 
                                          predictions_df.reset_index(drop=True)], axis=1)
            else:
                logger.warning("Input data length doesn't match predictions. Skipping input data inclusion.")
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.csv':
            predictions_df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in ['.xlsx', '.xls']:
            predictions_df.to_excel(output_path, index=False)
        else:
            # Default to CSV
            predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved successfully with {len(predictions_df)} rows")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary of feature importance scores or None if not available
        """
        if not self.is_ready:
            return None
        
        return self.model.get_feature_importance()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'is_ready': self.is_ready,
            'has_preprocessor': self.data_processor.preprocessor is not None
        }
        
        if self.is_ready:
            info.update(self.model.get_model_info())
        
        return info
    
    def hyperparameter_tuning(self, training_data_path: str, param_grid: Dict = None,
                            target_column: str = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning on the model.
        
        Args:
            training_data_path: Path to training data
            param_grid: Parameter grid for tuning
            target_column: Target column name
            
        Returns:
            Dictionary containing tuning results
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Load and preprocess data
        X, y = self.data_processor.load_and_preprocess(training_data_path, target_column)
        
        # Perform tuning
        tuning_results = self.model.hyperparameter_tuning(X, y, param_grid)
        
        # Mark as ready since model is now trained
        self.is_ready = True
        
        return tuning_results
    
    def batch_predict(self, input_directory: str, output_directory: str, 
                     file_pattern: str = "*.csv", include_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions on multiple files in a directory.
        
        Args:
            input_directory: Directory containing input files
            output_directory: Directory to save prediction files
            file_pattern: Pattern to match input files
            include_probabilities: Whether to include probabilities in output
            
        Returns:
            Dictionary containing batch processing results
        """
        if not self.is_ready:
            raise ValueError("Predictor not ready. Load a model or train first.")
        
        input_path = Path(input_directory)
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        input_files = list(input_path.glob(file_pattern))
        
        if not input_files:
            logger.warning(f"No files found matching pattern '{file_pattern}' in {input_directory}")
            return {'processed_files': 0, 'results': []}
        
        logger.info(f"Processing {len(input_files)} files...")
        
        batch_results = []
        
        for input_file in input_files:
            try:
                logger.info(f"Processing: {input_file.name}")
                
                # Make predictions
                result = self.predict_from_file(str(input_file), include_probabilities)
                
                # Save predictions
                output_file = output_path / f"{input_file.stem}_predictions{input_file.suffix}"
                
                # Load original data for saving with predictions
                original_data = self.data_processor.load_data(str(input_file))
                self.save_predictions(result, str(output_file), 
                                    include_input_data=True, input_data=original_data)
                
                batch_results.append({
                    'input_file': str(input_file),
                    'output_file': str(output_file),
                    'predictions_count': result['prediction_count'],
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {str(e)}")
                batch_results.append({
                    'input_file': str(input_file),
                    'output_file': None,
                    'predictions_count': 0,
                    'status': 'error',
                    'error': str(e)
                })
        
        successful_files = sum(1 for r in batch_results if r['status'] == 'success')
        logger.info(f"Batch processing completed. {successful_files}/{len(input_files)} files processed successfully.")
        
        return {
            'processed_files': len(input_files),
            'successful_files': successful_files,
            'results': batch_results
        }
    
    def __str__(self) -> str:
        """String representation of the predictor."""
        status = "ready" if self.is_ready else "not ready"
        return f"SmokeFreePredictor(status={status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the predictor."""
        return f"SmokeFreePredictor({self.get_model_info()})"