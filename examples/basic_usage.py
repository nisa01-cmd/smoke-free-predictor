#!/usr/bin/env python3
"""
Basic Usage Example - Smoke-Free Predictor
==========================================

This script demonstrates the basic usage of the Smoke-Free Predictor
including training a model, making predictions, and evaluating results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from predictor import SmokeFreePredictor
from predictor.model import SmokeFreeModel
from predictor.data_processor import DataProcessor


def create_sample_data():
    """Create sample data for demonstration purposes."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    
    data = {
        'participant_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'smoking_years': np.random.normal(20, 10, n_samples).astype(int),
        'cigarettes_per_day': np.random.normal(15, 8, n_samples).astype(int),
        'quit_attempts': np.random.poisson(2, n_samples),
        'support_system': np.random.choice(['low', 'medium', 'high'], n_samples),
        'motivation_score': np.random.uniform(1, 10, n_samples),
        'health_concerns': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'financial_motivation': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    }
    
    # Create outcome based on features (with some randomness)
    outcome_prob = (
        0.1 +  # baseline
        0.2 * (data['motivation_score'] / 10) +
        0.15 * data['health_concerns'] +
        0.1 * data['financial_motivation'] +
        0.1 * (data['support_system'] == 'high').astype(int) +
        0.05 * (data['support_system'] == 'medium').astype(int) -
        0.02 * (data['smoking_years'] / 30) -
        0.01 * (data['cigarettes_per_day'] / 20)
    )
    
    # Add some noise and create binary outcome
    outcome_prob += np.random.normal(0, 0.1, n_samples)
    data['smoke_free_outcome'] = (outcome_prob > 0.5).astype(int)
    
    # Clean up unrealistic values
    data['age'] = np.clip(data['age'], 18, 80)
    data['smoking_years'] = np.clip(data['smoking_years'], 1, data['age'] - 18)
    data['cigarettes_per_day'] = np.clip(data['cigarettes_per_day'], 1, 40)
    data['quit_attempts'] = np.clip(data['quit_attempts'], 0, 10)
    
    return pd.DataFrame(data)

def main():
    """Main demonstration function."""
    print("🚬 Smoke-Free Predictor - Basic Usage Example")
    print("=" * 50)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    # Step 1: Create sample data
    print("\n1. Creating sample data...")
    df = create_sample_data()
    
    # Split into train and test
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    # Save datasets
    train_data.to_csv("data/sample_training_data.csv", index=False)
    test_data.to_csv("data/sample_test_data.csv", index=False)
    
    print(f"   Created {len(train_data)} training samples")
    print(f"   Created {len(test_data)} test samples")
    print(f"   Success rate in training data: {train_data['smoke_free_outcome'].mean():.2%}")
    
    # Step 2: Train a model
    print("\n2. Training the model...")
    predictor = SmokeFreePredictor()
    
    training_results = predictor.train(
        training_data_path="data/sample_training_data.csv",
        validation_split=0.2,
        save_model=True,
        model_output_path="models/sample_model.joblib"
    )
    
    print("   Training completed!")
    print(f"   Model type: {training_results['model_type']}")
    print(f"   Training accuracy: {training_results['train_metrics'].get('accuracy', 'N/A'):.4f}")
    if training_results['validation_metrics']:
        print(f"   Validation accuracy: {training_results['validation_metrics'].get('accuracy', 'N/A'):.4f}")
    
    # Step 3: Make predictions
    print("\n3. Making predictions...")
    prediction_results = predictor.predict_from_file(
        "data/sample_test_data.csv", 
        include_probabilities=True
    )
    
    print(f"   Made predictions for {prediction_results['prediction_count']} samples")
    
    # Save predictions
    test_data_for_pred = test_data.drop(columns=['smoke_free_outcome'])  # Remove target for prediction
    test_data_for_pred.to_csv("data/sample_prediction_input.csv", index=False)
    
    pred_input_results = predictor.predict_from_file(
        "data/sample_prediction_input.csv",
        include_probabilities=True
    )
    
    predictor.save_predictions(
        pred_input_results, 
        "outputs/sample_predictions.csv",
        include_input_data=True,
        input_data=test_data_for_pred
    )
    
    print("   Predictions saved to outputs/sample_predictions.csv")
    
    # Step 4: Evaluate the model
    print("\n4. Evaluating the model...")
    evaluation_results = predictor.evaluate_from_file("data/sample_test_data.csv")
    
    print("   Evaluation Results:")
    for metric, value in evaluation_results['metrics'].items():
        if isinstance(value, float):
            print(f"   - {metric.capitalize()}: {value:.4f}")
    
    # Step 5: Feature importance
    print("\n5. Feature importance...")
    importance = predictor.get_feature_importance()
    if importance:
        print("   Top 5 most important features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_importance[:5]:
            print(f"   - {feature}: {score:.4f}")
    
    # Step 6: Single prediction example
    print("\n6. Single prediction example...")
    sample_person = {
        'age': 35,
        'gender': 'F',
        'smoking_years': 10,
        'cigarettes_per_day': 12,
        'quit_attempts': 2,
        'support_system': 'high',
        'motivation_score': 8.5,
        'health_concerns': 1,
        'financial_motivation': 1
    }
    
    single_result = predictor.predict_single(sample_person, include_probabilities=True)
    
    print("   Sample person profile:")
    for key, value in sample_person.items():
        print(f"   - {key}: {value}")
    
    print(f"   Prediction: {'Success' if single_result['prediction'] == 1 else 'Not successful'}")
    if 'probabilities' in single_result:
        prob_success = single_result['probabilities'][1] if len(single_result['probabilities']) > 1 else single_result['probabilities'][0]
        print(f"   Probability of success: {prob_success:.2%}")
    
    print("\n✅ Example completed successfully!")
    print("\nFiles created:")
    print("- data/sample_training_data.csv")
    print("- data/sample_test_data.csv")
    print("- models/sample_model.joblib")
    print("- outputs/sample_predictions.csv")
    
    print("\n💡 Next steps:")
    print("- Try the CLI: python main.py --help")
    print("- Explore hyperparameter tuning")
    print("- Use your own data")
    print("- Check out other examples in this directory")
