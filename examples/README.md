# Examples

This directory contains example scripts and notebooks demonstrating how to use the Smoke-Free Predictor.

## Available Examples

### `basic_usage.py`
A comprehensive example showing:
- How to create sample data
- Training a model
- Making predictions  
- Evaluating model performance
- Feature importance analysis
- Single prediction examples

Run with:
```bash
cd examples
python basic_usage.py
```

## Coming Soon

- **Jupyter Notebooks**: Interactive examples with visualizations
- **Advanced Usage**: Hyperparameter tuning, custom preprocessing
- **Data Analysis**: Exploratory data analysis examples
- **Visualization**: Charts and plots for model interpretation
- **API Usage**: Examples using the Python API directly

## Sample Data

The examples generate synthetic data for demonstration purposes. The data includes:

- **Demographics**: Age, gender
- **Smoking History**: Years smoking, cigarettes per day, quit attempts
- **Support Factors**: Support system level, motivation score
- **Motivations**: Health concerns, financial motivation
- **Outcome**: Smoke-free success (binary)

## Running Examples

Make sure you have installed the requirements:
```bash
pip install -r ../requirements.txt
```

Then run any example from within the examples directory:
```bash
cd examples
python basic_usage.py
```

The examples will create sample data files in the `data/` directory and save results in `outputs/` and `models/`.