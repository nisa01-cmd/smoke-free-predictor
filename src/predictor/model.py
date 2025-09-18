"""
Machine Learning Model Module
============================

This module contains the machine learning model implementations for 
the Smoke-Free Predictor application.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import config

logger = logging.getLogger(__name__)


class SmokeFreeModel:
    """
    Main machine learning model class for smoke-free behavior prediction.
    
    This class handles both classification and regression tasks depending on 
    the target variable type.
    """
    
    def __init__(self, model_type: str = 'auto', **model_params):
        """
        Initialize the smoke-free prediction model.
        
        Args:
            model_type: Type of model ('classification', 'regression', 'auto')
            **model_params: Additional parameters for the underlying model
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_classes = None
        self.model_params = model_params
        self.training_history = {}
        
        # Use configuration parameters as defaults
        self.default_params = {
            'n_estimators': config.model.n_estimators,
            'max_depth': config.model.max_depth,
            'min_samples_split': config.model.min_samples_split,
            'min_samples_leaf': config.model.min_samples_leaf,
            'random_state': config.model.random_state
        }
        
        # Merge with any provided parameters
        self.default_params.update(model_params)
        
    def _determine_model_type(self, y: np.ndarray) -> str:
        """
        Automatically determine if this is a classification or regression task.
        
        Args:
            y: Target variable array
            
        Returns:
            'classification' or 'regression'
        """
        # Check if target is continuous or discrete
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        # If there are very few unique values relative to total, treat as classification
        if unique_values <= 10 or unique_values / total_values < 0.05:
            return 'classification'
        else:
            return 'regression'
    
    def _create_model(self, model_type: str) -> BaseEstimator:
        """
        Create the appropriate model based on type.
        
        Args:
            model_type: 'classification' or 'regression'
            
        Returns:
            Initialized model instance
        """
        if model_type == 'classification':
            return RandomForestClassifier(**self.default_params)
        elif model_type == 'regression':
            return RandomForestRegressor(**self.default_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training metrics and information
        """
        logger.info("Starting model training...")
        
        # Determine model type if set to auto
        if self.model_type == 'auto':
            self.model_type = self._determine_model_type(y)
            logger.info(f"Auto-determined model type: {self.model_type}")
        
        # Create the model
        self.model = self._create_model(self.model_type)
        
        # Split data if validation is requested
        validation_split = validation_split or config.model.validation_split
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                random_state=config.model.random_state,
                stratify=y if self.model_type == 'classification' else None
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Train the model
        logger.info(f"Training {self.model_type} model on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        
        # Store training information
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if self.model_type == 'classification':
            self.target_classes = self.model.classes_
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X_train, y_train, 'training')
        
        # Calculate validation metrics if validation set exists
        val_metrics = {}
        if X_val is not None:
            val_metrics = self._calculate_metrics(X_val, y_val, 'validation')
        
        # Perform cross-validation
        cv_scores = self._cross_validate(X, y)
        
        # Store training history
        self.training_history = {
            'model_type': self.model_type,
            'training_samples': X_train.shape[0],
            'features': X_train.shape[1],
            'validation_split': validation_split,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'cross_validation': cv_scores,
            'model_params': self.default_params
        }
        
        logger.info("Model training completed successfully!")
        logger.info(f"Training {self._get_primary_metric()}: {train_metrics.get(self._get_primary_metric(), 'N/A'):.4f}")
        
        if val_metrics:
            logger.info(f"Validation {self._get_primary_metric()}: {val_metrics.get(self._get_primary_metric(), 'N/A'):.4f}")
        
        return self.training_history
    
    def _get_primary_metric(self) -> str:
        """Get the primary metric name based on model type."""
        return 'accuracy' if self.model_type == 'classification' else 'r2'
    
    def _calculate_metrics(self, X: np.ndarray, y_true: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """
        Calculate performance metrics for the model.
        
        Args:
            X: Feature matrix
            y_true: True target values
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary of calculated metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating metrics")
        
        y_pred = self.model.predict(X)
        metrics = {}
        
        if self.model_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle binary vs multiclass classification
            average_method = 'binary' if len(self.target_classes) == 2 else 'weighted'
            
            metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)
            
            # ROC AUC for binary classification or if predict_proba is available
            if len(self.target_classes) == 2 and hasattr(self.model, 'predict_proba'):
                try:
                    y_proba = self.model.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                except:
                    pass  # Skip if unable to calculate
                    
        else:  # regression
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of cross-validation scores
        """
        if not self.is_trained:
            return {}
        
        logger.info("Performing cross-validation...")
        
        # Determine scoring metric
        scoring = 'accuracy' if self.model_type == 'classification' else 'r2'
        
        try:
            cv_scores = cross_val_score(
                self.model, X, y, 
                cv=config.model.cross_validation_folds,
                scoring=scoring
            )
            
            return {
                f'cv_{scoring}_mean': cv_scores.mean(),
                f'cv_{scoring}_std': cv_scores.std(),
                f'cv_{scoring}_scores': cv_scores.tolist()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Prediction probabilities
            
        Raises:
            ValueError: If model hasn't been trained or doesn't support probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type != 'classification':
            raise ValueError("Prediction probabilities only available for classification models")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return None
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_scores = self.model.feature_importances_
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance_scores))]
        
        return dict(zip(feature_names, importance_scores))
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target values
            
        Returns:
            Dictionary containing evaluation metrics and results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model on test data...")
        
        # Calculate basic metrics
        test_metrics = self._calculate_metrics(X_test, y_test, 'test')
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        evaluation_results = {
            'metrics': test_metrics,
            'predictions': y_pred.tolist(),
            'true_values': y_test.tolist()
        }
        
        # Add classification-specific results
        if self.model_type == 'classification':
            evaluation_results['classification_report'] = classification_report(
                y_test, y_pred, output_dict=True
            )
            evaluation_results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
            # Add probabilities if available
            if hasattr(self.model, 'predict_proba'):
                evaluation_results['prediction_probabilities'] = self.predict_proba(X_test).tolist()
        
        # Add feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance:
            evaluation_results['feature_importance'] = feature_importance
        
        logger.info(f"Evaluation completed. Test {self._get_primary_metric()}: {test_metrics.get(self._get_primary_metric(), 'N/A'):.4f}")
        
        return evaluation_results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, param_grid: Dict = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for tuning (uses default if None)
            
        Returns:
            Dictionary containing tuning results
        """
        logger.info("Starting hyperparameter tuning...")
        
        if param_grid is None:
            # Default parameter grid
            if self.model_type == 'classification':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
        
        # Create base model
        base_model = self._create_model(self.model_type)
        
        # Perform grid search
        scoring = 'accuracy' if self.model_type == 'classification' else 'r2'
        
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=config.model.cross_validation_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.default_params.update(grid_search.best_params_)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed. Best {scoring}: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return tuning_results
    
    def save(self, file_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Ensure the directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history,
            'model_params': self.default_params,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to: {file_path}")
    
    def load(self, file_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to load the model from
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data.get('feature_names')
            self.target_classes = model_data.get('target_classes')
            self.training_history = model_data.get('training_history', {})
            self.default_params = model_data.get('model_params', {})
            self.is_trained = model_data.get('is_trained', True)
            
            logger.info(f"Model loaded from: {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'model_class': type(self.model).__name__ if self.model else None,
            'parameters': self.default_params,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'training_history': self.training_history
        }
        
        if self.model_type == 'classification' and self.target_classes is not None:
            info['target_classes'] = self.target_classes.tolist()
            
        return info
    
    def __str__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"SmokeFreeModel(type={self.model_type}, status={status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"SmokeFreeModel({self.get_model_info()})"