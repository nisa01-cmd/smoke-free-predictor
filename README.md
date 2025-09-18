# Smoke-Free Predictor

A comprehensive machine learning application for predicting smoke-free behavior outcomes. This project provides tools for training, evaluating, and deploying predictive models to analyze factors that contribute to successful smoke cessation.

## 🎯 Overview

The Smoke-Free Predictor uses machine learning algorithms to analyze various factors and predict the likelihood of successful smoke-free outcomes. The project includes data preprocessing, model training, evaluation tools, and a user-friendly prediction interface.

## ✨ Features

- **Automated Data Processing**: Clean, preprocess, and engineer features from raw data
- **Multiple ML Algorithms**: Support for classification and regression tasks with automatic model type detection
- **Comprehensive Evaluation**: Detailed metrics, cross-validation, and performance analysis
- **Easy-to-use CLI**: Command-line interface for training, prediction, and evaluation
- **Batch Processing**: Process multiple files at once
- **Model Persistence**: Save and load trained models with preprocessing pipelines
- **Hyperparameter Tuning**: Automated parameter optimization
- **Flexible Data Formats**: Support for CSV, Excel, JSON, and Parquet files

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nisa01-cmd/smoke-free-predictor.git
cd smoke-free-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Training a Model
```bash
python main.py train --data data/training_data.csv
```

#### Making Predictions
```bash
python main.py predict --input data/new_data.csv --output predictions.csv
```

#### Evaluating a Model
```bash
python main.py evaluate --test-data data/test_data.csv
```

## 📁 Project Structure

```
smoke-free-predictor/
├── main.py                 # CLI entry point
├── src/                    # Core package
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── data_processor.py  # Data preprocessing
│   ├── model.py           # ML model implementation
│   └── predictor.py       # High-level prediction interface
├── data/                  # Data directory
├── models/                # Saved models
├── outputs/               # Prediction outputs
├── tests/                 # Test files
├── examples/              # Example scripts and notebooks
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🛠️ Configuration

The application uses a configuration system that supports:
- Default settings via `src/config.py`
- Environment variable overrides
- Command-line parameter overrides

### Key Configuration Options

```python
# Model parameters
MODEL_N_ESTIMATORS=100
MODEL_MAX_DEPTH=None
MODEL_RANDOM_STATE=42

# Data processing
DATA_DIR=data
MODELS_DIR=models
OUTPUTS_DIR=outputs
TARGET_COLUMN=smoke_free_outcome

# Logging
LOG_LEVEL=INFO
```

## 📊 Data Format

The application expects data in tabular format with:
- **Features**: Columns containing predictor variables
- **Target**: Column containing the outcome to predict (specified in config)
- **Optional ID**: Column for participant/sample identification

### Example Data Structure
```csv
participant_id,age,gender,smoking_years,quit_attempts,support_system,smoke_free_outcome
1,45,M,20,3,high,1
2,32,F,8,1,medium,0
3,55,M,30,5,high,1
```

## 🤖 Model Types

The application automatically detects the appropriate model type:
- **Classification**: For discrete outcomes (success/failure, categories)
- **Regression**: For continuous outcomes (days smoke-free, reduction percentage)

Currently supports Random Forest algorithms with plans to add:
- Gradient Boosting
- Neural Networks
- Ensemble Methods

## 📈 Evaluation Metrics

### Classification
- Accuracy
- Precision, Recall, F1-score
- ROC AUC
- Confusion Matrix
- Classification Report

### Regression
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🔧 Advanced Usage

### Python API

```python
from src.predictor import SmokeFreePredictor

# Initialize predictor
predictor = SmokeFreePredictor()

# Train model
training_results = predictor.train('data/training_data.csv')

# Make predictions
predictions = predictor.predict_from_file('data/new_data.csv')

# Evaluate model
evaluation = predictor.evaluate_from_file('data/test_data.csv')
```

### Hyperparameter Tuning

```python
from src.model import SmokeFreeModel

model = SmokeFreeModel()
tuning_results = model.hyperparameter_tuning(X, y, {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
})
```

### Batch Processing

```python
# Process multiple files at once
results = predictor.batch_predict(
    input_directory='data/batch_input/',
    output_directory='outputs/batch_results/',
    file_pattern='*.csv'
)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📚 Examples

See the `examples/` directory for:
- Jupyter notebooks with detailed walkthroughs
- Sample datasets
- Common use case implementations
- Visualization examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/nisa01-cmd/smoke-free-predictor/issues) page
2. Create a new issue with detailed information
3. For urgent matters, contact the maintainers

## 🙏 Acknowledgments

- Built with scikit-learn, pandas, and numpy
- Inspired by public health research on smoking cessation
- Community contributions and feedback

## 📋 Changelog

### Version 1.0.0
- Initial release
- Basic prediction functionality
- CLI interface
- Data preprocessing pipeline
- Model training and evaluation

---

**Note**: This is a machine learning tool for research and analysis purposes. Always consult healthcare professionals for medical advice and smoking cessation programs.
