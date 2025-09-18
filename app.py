#!/usr/bin/env python3
"""
Flask Web Application for Smoke-Free Predictor
==============================================

Web interface and API for the smoke-free prediction model.
Designed for easy deployment to cloud platforms.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
import sys
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')
try:
    from predictor import SmokeFreePredictor
    from config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'smoke-free-predictor-secret-key-change-in-production')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

def init_predictor():
    """Initialize the predictor with a pre-trained model or create a demo model."""
    global predictor
    
    model_path = 'models/smoke_free_model.joblib'
    
    if Path(model_path).exists():
        logger.info("Loading existing model...")
        try:
            predictor = SmokeFreePredictor(model_path)
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            create_demo_model()
    else:
        logger.info("No existing model found, creating demo model...")
        create_demo_model()

def create_demo_model():
    """Create a demo model with synthetic data for deployment."""
    global predictor
    
    try:
        logger.info("Creating synthetic training data...")
        
        # Create demo data
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'age': np.random.normal(35, 12, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'stress_level': np.random.randint(1, 11, n_samples),
            'peer_pressure': np.random.randint(1, 11, n_samples),
            'exercise_freq': np.random.randint(0, 6, n_samples),
            'sleep_hours': np.random.normal(7, 1.5, n_samples),
            'motivation_score': np.random.uniform(1, 10, n_samples),
            'support_system': np.random.choice(['low', 'medium', 'high'], n_samples),
        }
        
        # Create realistic outcome based on features
        outcome_prob = (
            0.2 +  # baseline
            0.15 * (data['motivation_score'] / 10) +
            0.1 * (data['support_system'] == 'high').astype(int) +
            0.05 * (data['support_system'] == 'medium').astype(int) +
            0.1 * (data['exercise_freq'] / 5) +
            0.05 * (data['sleep_hours'] / 10) -
            0.1 * (data['stress_level'] / 10) -
            0.05 * (data['peer_pressure'] / 10)
        )
        
        outcome_prob += np.random.normal(0, 0.1, n_samples)
        data['smoke_free_outcome'] = (outcome_prob > 0.5).astype(int)
        
        # Clean up data
        data['age'] = np.clip(data['age'], 18, 80)
        data['sleep_hours'] = np.clip(data['sleep_hours'], 4, 12)
        
        # Save demo data
        Path('data').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        
        demo_df = pd.DataFrame(data)
        demo_df.to_csv('data/demo_training_data.csv', index=False)
        
        # Train the model
        predictor = SmokeFreePredictor()
        predictor.train(
            'data/demo_training_data.csv',
            save_model=True,
            model_output_path='models/smoke_free_model.joblib'
        )
        
        logger.info("✅ Demo model created and trained successfully!")
        
    except Exception as e:
        logger.error(f"Error creating demo model: {e}")
        logger.error(traceback.format_exc())
        # Create a minimal fallback
        predictor = SmokeFreePredictor()

# Routes
@app.route('/')
def home():
    """Home page with prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        data = {
            'age': int(request.form.get('age', 25)),
            'gender': request.form.get('gender', 'Male'),
            'stress_level': int(request.form.get('stress_level', 5)),
            'peer_pressure': int(request.form.get('peer_pressure', 5)),
            'exercise_freq': int(request.form.get('exercise_freq', 2)),
            'sleep_hours': float(request.form.get('sleep_hours', 7)),
            'motivation_score': float(request.form.get('motivation_score', 7)),
            'support_system': request.form.get('support_system', 'medium'),
        }
        
        # Make prediction
        if predictor and predictor.is_ready:
            result = predictor.predict_single(data, include_probabilities=True)
            
            # Format result for display
            prediction = result['prediction']
            prediction_text = "Success" if prediction == 1 else "Not Successful"
            
            probability = None
            if 'probabilities' in result and result['probabilities']:
                if len(result['probabilities']) > 1:
                    probability = result['probabilities'][1]  # Probability of success
                else:
                    probability = result['probabilities'][0]
            
            return render_template('result.html', 
                                 prediction=prediction,
                                 prediction_text=prediction_text,
                                 probability=probability,
                                 input_data=data)
        else:
            flash('Model not available. Please try again later.', 'error')
            return redirect(url_for('predict'))
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('predict'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        if not predictor or not predictor.is_ready:
            return jsonify({
                'error': 'Model not available',
                'status': 'error'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        # Validate required fields
        required_fields = ['age', 'gender', 'stress_level', 'peer_pressure', 
                          'exercise_freq', 'sleep_hours']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = predictor.predict_single(data, include_probabilities=True)
        
        # Return formatted result
        response = {
            'prediction': int(result['prediction']),
            'prediction_text': "Success" if result['prediction'] == 1 else "Not Successful",
            'model_type': result['model_type'],
            'input_data': result['input_features'],
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        if 'probabilities' in result:
            response['probabilities'] = result['probabilities']
            if len(result['probabilities']) > 1:
                response['success_probability'] = result['probabilities'][1]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_ready': predictor is not None and predictor.is_ready,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info')
def model_info():
    """Get model information."""
    if predictor and predictor.is_ready:
        info = predictor.get_model_info()
        return jsonify({
            'status': 'success',
            'model_info': info,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Model not available'
        }), 500

@app.route('/batch-predict', methods=['GET', 'POST'])
def batch_predict():
    """Handle batch predictions via file upload."""
    if request.method == 'GET':
        return render_template('batch.html')
    
    try:
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('batch_predict'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('batch_predict'))
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = Path('uploads') / filename
            filepath.parent.mkdir(exist_ok=True)
            
            file.save(str(filepath))
            
            # Make predictions
            if predictor and predictor.is_ready:
                results = predictor.predict_from_file(str(filepath), include_probabilities=True)
                
                # Save results
                output_path = Path('outputs') / f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                output_path.parent.mkdir(exist_ok=True)
                
                # Load original data for combining
                original_data = pd.read_csv(filepath)
                predictor.save_predictions(results, str(output_path), 
                                         include_input_data=True, input_data=original_data)
                
                flash(f'Batch predictions completed! {results["prediction_count"]} predictions made.', 'success')
                return render_template('batch_result.html', 
                                     results=results, 
                                     output_file=output_path.name)
            else:
                flash('Model not available', 'error')
                return redirect(url_for('batch_predict'))
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(url_for('batch_predict'))
            
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('batch_predict'))

@app.route('/about')
def about():
    """About page with model information."""
    model_info = None
    if predictor and predictor.is_ready:
        model_info = predictor.get_model_info()
    
    return render_template('about.html', model_info=model_info)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

# Initialize the app
if __name__ == '__main__':
    # Create necessary directories
    Path('uploads').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    
    # Initialize predictor
    init_predictor()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
else:
    # For production deployment
    init_predictor()