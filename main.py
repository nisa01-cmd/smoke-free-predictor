#!/usr/bin/env python3
"""
Smoke-Free Predictor - Main Application
======================================

A machine learning application to predict smoke-free behavior and outcomes.
This main module provides the CLI interface and orchestrates the prediction pipeline.

Usage:
    python main.py --help
    python main.py train --data data/training_data.csv
    python main.py predict --input data/new_data.csv --output predictions.csv
    python main.py evaluate --test-data data/test_data.csv
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import pandas as pd 
from src.predictor import SmokeFreePredictor
from src.predictor.data_processor import DataProcessor
from src.predictor.model import SmokeFreeModel
from src.predictor.config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_arguments() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Smoke-Free Predictor - Predict smoke-free behavior outcomes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    train_parser.add_argument('--data', '-d', required=True, 
                            help='Path to training data CSV file')
    train_parser.add_argument('--model-output', '-o', 
                            default='models/smoke_free_model.joblib',
                            help='Path to save the trained model')
    train_parser.add_argument('--validation-split', type=float, default=0.2,
                            help='Fraction of data to use for validation')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new data')
    predict_parser.add_argument('--input', '-i', required=True,
                              help='Path to input data CSV file')
    predict_parser.add_argument('--output', '-o', required=True,
                              help='Path to save predictions CSV file')
    predict_parser.add_argument('--model', '-m',
                              default='models/smoke_free_model.joblib',
                              help='Path to trained model file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--test-data', '-t', required=True,
                               help='Path to test data CSV file')
    evaluate_parser.add_argument('--model', '-m',
                               default='models/smoke_free_model.joblib',
                               help='Path to trained model file')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    return parser


def train_model(args) -> None:
    """Train the smoke-free prediction model."""
    logger.info(f"Training model with data from: {args.data}")

    try:
        # Load and process data
        data_processor = DataProcessor()
        X, y = data_processor.load_and_preprocess(args.data)

        # Create and train model
        model = SmokeFreeModel()
        model.train(X, y, validation_split=args.validation_split)

        # Corrected part:
        # Save the trained model using the model's own save method.
        # This will save all necessary metadata.
        model.save(args.model_output)

        # Also save the preprocessor separately. This is a common and robust
        # pattern in machine learning pipelines.
        preprocessor_path = Path(args.model_output).with_suffix('.preprocessor.joblib')
        data_processor.save_preprocessor(str(preprocessor_path))

        logger.info(f"Model saved to: {args.model_output}")
        logger.info(f"Preprocessor saved to: {preprocessor_path}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)

def make_predictions(args) -> None:
    """Make predictions on new data."""
    logger.info(f"Making predictions on data from: {args.input}")

    try:
        # Initialize the model and data processor
        model = SmokeFreeModel()
        data_processor = DataProcessor()

        # Load the model from its dedicated file
        model_path = args.model
        model.load(model_path)

        # Load the preprocessor from its dedicated file
        preprocessor_path = Path(model_path).with_suffix('.preprocessor.joblib')
        data_processor.load_preprocessor(str(preprocessor_path))
        
        # Load new data for prediction
        input_data = data_processor.load_data(args.input)
        
        # Transform the data using the loaded preprocessor
        X_transformed = data_processor.transform_new_data(input_data)
        
        # Make predictions using the loaded model
        predictions = model.predict(X_transformed)

        # Save predictions (this part seems fine from your provided code)
        result = pd.DataFrame(predictions, columns=['prediction'])
        result['prediction_label'] = data_processor._target_encoder.inverse_transform(predictions)
        input_data['prediction'] = result['prediction_label']
        input_data.to_csv(args.output, index=False)
        
        logger.info(f"Predictions saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        sys.exit(1)

def evaluate_model(args) -> None:
    """Evaluate model performance on test data."""
    logger.info(f"Evaluating model on test data from: {args.test_data}")
    
    try:
        # Initialize predictor
        predictor = SmokeFreePredictor(args.model)
        
        # Evaluate model
        metrics = predictor.evaluate_from_file(args.test_data)
        
        # Display results
        print("\nModel Evaluation Results:")
        print("=" * 25)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)


def show_version() -> None:
    """Display version information."""
    print("Smoke-Free Predictor v1.0.0")
    print("A machine learning application for predicting smoke-free behavior outcomes")


def main():
    """Main application entry point."""
    parser = setup_arguments()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        make_predictions(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'version':
        show_version()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()