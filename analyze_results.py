#!/usr/bin/env python3
"""
Results Analysis Script - Smoke-Free Predictor
==============================================

This script provides comprehensive analysis of your model's performance
and predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src directory to path
sys.path.append("src")
from predictor import SmokeFreePredictor

def analyze_predictions():
    """Analyze the predictions made by the model."""
    print("📊 PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Load predictions
    if Path("outputs/predictions.csv").exists():
        predictions_df = pd.read_csv("outputs/predictions.csv")
        print(f"✅ Found {len(predictions_df)} predictions")
        
        # Basic stats
        print(f"\nPrediction Summary:")
        print(f"- Success predictions (1): {(predictions_df['prediction'] == 1).sum()} ({(predictions_df['prediction'] == 1).mean():.1%})")
        print(f"- Failure predictions (0): {(predictions_df['prediction'] == 0).sum()} ({(predictions_df['prediction'] == 0).mean():.1%})")
        
        # Show sample predictions
        print(f"\n📋 Sample Predictions:")
        print(predictions_df.head())
        
        return predictions_df
    else:
        print("❌ No predictions found. Run predictions first!")
        return None

def analyze_training_data():
    """Analyze the training data."""
    print(f"\n📚 TRAINING DATA ANALYSIS") 
    print("=" * 50)
    
    if Path("data/training_data.csv").exists():
        train_df = pd.read_csv("data/training_data.csv")
        print(f"✅ Training data: {len(train_df)} samples")
        
        # Check if there's a target column
        possible_targets = ['smoke_free_outcome', 'outcome', 'target', 'success']
        target_col = None
        
        for col in possible_targets:
            if col in train_df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"🎯 Target column: '{target_col}'")
            print(f"- Success rate: {train_df[target_col].mean():.1%}")
            print(f"- Class distribution:")
            print(train_df[target_col].value_counts())
        
        print(f"\n📊 Data Overview:")
        print(train_df.info())
        print(f"\n📈 Statistical Summary:")
        print(train_df.describe())
        
        return train_df, target_col
    else:
        print("❌ No training data found")
        return None, None

def evaluate_model():
    """Evaluate model performance if we have test data."""
    print(f"\n🎯 MODEL EVALUATION")
    print("=" * 50)
    
    try:
        # Check if we have a saved model
        if Path("models/smoke_free_model.joblib").exists():
            predictor = SmokeFreePredictor("models/smoke_free_model.joblib")
            
            # Get model info
            info = predictor.get_model_info()
            print(f"✅ Model loaded successfully")
            print(f"- Model type: {info.get('model_type', 'Unknown')}")
            print(f"- Training status: {'Trained' if info.get('is_trained', False) else 'Not trained'}")
            print(f"- Features: {info.get('feature_count', 'Unknown')}")
            
            # Try to evaluate on training data (as a sanity check)
            if Path("data/training_data.csv").exists():
                try:
                    results = predictor.evaluate_from_file("data/training_data.csv")
                    print(f"\n📈 Performance on training data:")
                    for metric, value in results['metrics'].items():
                        if isinstance(value, (int, float)):
                            print(f"- {metric.capitalize()}: {value:.4f}")
                except Exception as e:
                    print(f"⚠️ Could not evaluate: {e}")
            
            # Feature importance
            importance = predictor.get_feature_importance()
            if importance:
                print(f"\n🏆 Top Feature Importance:")
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, score in sorted_features[:5]:
                    print(f"- {feature}: {score:.4f}")
            
            return predictor
        else:
            print("❌ No saved model found")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def create_visualization():
    """Create visualizations of the results."""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Create outputs directory for plots
        Path("outputs/plots").mkdir(exist_ok=True)
        
        # Load and visualize predictions
        if Path("outputs/predictions.csv").exists():
            df = pd.read_csv("outputs/predictions.csv")
            
            # Prediction distribution
            plt.figure(figsize=(10, 6))
            
            plt.subplot(2, 2, 1)
            df['prediction'].value_counts().plot(kind='bar', color=['red', 'green'])
            plt.title('Prediction Distribution')
            plt.xlabel('Prediction (0=Failure, 1=Success)')
            plt.ylabel('Count')
            
            # If we have numeric features, show their distributions
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'prediction']
            
            if len(numeric_cols) >= 1:
                plt.subplot(2, 2, 2)
                df[numeric_cols[0]].hist(bins=20, alpha=0.7)
                plt.title(f'{numeric_cols[0]} Distribution')
            
            if len(numeric_cols) >= 2:
                plt.subplot(2, 2, 3)
                scatter_colors = ['red' if p == 0 else 'green' for p in df['prediction']]
                plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], c=scatter_colors, alpha=0.6)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]} by Prediction')
            
            plt.tight_layout()
            plt.savefig("outputs/plots/prediction_analysis.png", dpi=150, bbox_inches='tight')
            plt.show()
            print("✅ Visualizations saved to outputs/plots/")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        print("   (This might be due to display issues in your environment)")

def suggest_improvements():
    """Suggest next steps for improvement."""
    print(f"\n💡 SUGGESTED IMPROVEMENTS")
    print("=" * 50)
    
    suggestions = [
        "🔍 **Data Quality**:",
        "   - Add more training data if possible",
        "   - Check for class imbalance in your target variable",
        "   - Verify that feature names match real-world smoking cessation factors",
        "",
        "🧠 **Model Enhancement**:",
        "   - Try hyperparameter tuning: python -c \"from src.predictor import SmokeFreePredictor; p = SmokeFreePredictor(); p.hyperparameter_tuning('data/training_data.csv')\"",
        "   - Experiment with different algorithms (coming in future updates)",
        "   - Add feature engineering (interaction terms, polynomial features)",
        "",
        "📊 **Validation**:",
        "   - Create a separate test set to validate true performance",
        "   - Use cross-validation for more robust evaluation",
        "   - Compare predictions with domain expert knowledge",
        "",
        "🚀 **Production Readiness**:",
        "   - Set up model versioning",
        "   - Create API endpoints for real-time predictions",
        "   - Add monitoring and logging for production use",
        "",
        "📈 **Analysis & Insights**:",
        "   - Analyze which factors are most predictive",
        "   - Create patient profiles for different risk levels",
        "   - Generate actionable insights for intervention programs"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """Main analysis function."""
    print("🚬 SMOKE-FREE PREDICTOR - RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Run all analyses
    predictions_df = analyze_predictions()
    train_df, target_col = analyze_training_data()
    predictor = evaluate_model()
    create_visualization()
    suggest_improvements()
    
    print(f"\n🎯 QUICK ACTIONS YOU CAN TRY RIGHT NOW:")
    print("=" * 50)
    print("1. 📊 View your predictions: notepad outputs/predictions.csv")
    print("2. 🔧 Tune hyperparameters: python main.py train --data data/training_data.csv --validation-split 0.3")
    print("3. 🧪 Test with new data: python main.py predict --input data/new_data.csv --output outputs/new_predictions.csv")
    print("4. 📈 Evaluate performance: python main.py evaluate --test-data data/training_data.csv")
    print("5. 🎨 Run visualization: python analyze_results.py")
    
    print(f"\n✨ Your model is working! Keep experimenting and improving! ✨")

if __name__ == "__main__":
    main()