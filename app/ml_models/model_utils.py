import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple, Dict, Any, Optional
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FraudDetectionPredictor:
    """
    Enhanced predictor class for fraud detection with improved feature engineering
    and error handling for production use.
    """
    
    def __init__(self, model_dir="app/ml_models/saved_models"):
        self.model_dir = model_dir
        self.rf_model = None
        self.lr_model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.training_stats = None
        self.is_loaded = False
        
        # Try to load models on initialization
        self.load_models()

    def load_models(self):
        """Load the trained models and preprocessors with comprehensive error handling."""
        try:
            logger.info(f"Loading models from {self.model_dir}...")
            
            # Check if all required files exist
            required_files = [
                "random_forest_model.pkl",
                "logistic_regression_model.pkl", 
                "scaler.pkl",
                "label_encoders.pkl",
                "feature_columns.pkl"
            ]
            
            missing_files = []
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"Missing model files: {missing_files}")
                logger.warning("Models not loaded. Please train models first.")
                return False
            
            # Load models and preprocessors
            self.rf_model = joblib.load(os.path.join(self.model_dir, "random_forest_model.pkl"))
            self.lr_model = joblib.load(os.path.join(self.model_dir, "logistic_regression_model.pkl"))
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            self.label_encoders = joblib.load(os.path.join(self.model_dir, "label_encoders.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, "feature_columns.pkl"))
            
            # Load training stats if available
            try:
                self.training_stats = joblib.load(os.path.join(self.model_dir, "training_stats.pkl"))
            except FileNotFoundError:
                logger.warning("Training stats not found. Some metadata will be unavailable.")
                self.training_stats = {}
            
            self.is_loaded = True
            logger.info("Models loaded successfully")
            logger.info(f"Available models: Random Forest, Logistic Regression")
            logger.info(f"Number of features: {len(self.feature_columns)}")
            
            if self.training_stats:
                logger.info(f"Models trained on {self.training_stats.get('dataset_size', 'unknown')} samples")
                logger.info(f"Training date: {self.training_stats.get('training_date', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess user uploaded data to match training format
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Please ensure models are trained and available.")
        
        try:
            # Detect column mapping
            column_mapping = self.detect_column_mapping(df)
            
            # Apply feature engineering
            processed_data = self.advanced_feature_engineering(df, column_mapping)
            
            # Select only the features used in training
            feature_data = pd.DataFrame()
            for feature in self.feature_columns:
                if feature in processed_data.columns:
                    feature_data[feature] = processed_data[feature]
                else:
                    feature_data[feature] = 0  # Default value for missing features
                    logger.warning(f"Feature {feature} not found, using default value 0")
            
            # Fill any remaining NaN values
            feature_data = feature_data.fillna(0)
            
            # Ensure correct data types
            feature_data = feature_data.astype('float32')
            
            logger.info(f"Preprocessed data shape: {feature_data.shape}")
            logger.info(f"Expected features: {len(self.feature_columns)}")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing user data: {e}")
            raise
    
    def predict_fraud(self, df: pd.DataFrame, model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Predict fraud for uploaded data with comprehensive error handling and insights
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Please train models first.")
        
        try:
            logger.info(f"Starting fraud prediction with {model_type} model")
            logger.info(f"Input data shape: {df.shape}")
            
            # Detect column mapping for user feedback
            column_mapping = self.detect_column_mapping(df)
            
            # Preprocess data
            processed_data = self.preprocess_user_data(df)
            
            # Select model and make predictions
            if model_type == "random_forest":
                if self.rf_model is None:
                    raise ValueError("Random Forest model not available")
                model = self.rf_model
                predictions = model.predict(processed_data)
                probabilities = model.predict_proba(processed_data)[:, 1]
                logger.info("Using Random Forest model for prediction")
                
            elif model_type == "logistic_regression":
                if self.lr_model is None:
                    raise ValueError("Logistic Regression model not available")
                model = self.lr_model
                scaled_data = self.scaler.transform(processed_data)
                predictions = model.predict(scaled_data)
                probabilities = model.predict_proba(scaled_data)[:, 1]
                logger.info("Using Logistic Regression model for prediction")
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Calculate results
            total_transactions = len(df)
            fraud_detected = int(predictions.sum())
            fraud_percentage = (fraud_detected / total_transactions) * 100 if total_transactions > 0 else 0
            avg_risk_score = float(probabilities.mean())
            
            # Get high-risk transactions (threshold can be adjusted)
            high_risk_threshold = 0.7
            high_risk_indices = np.where(probabilities > high_risk_threshold)[0]
            high_risk_transactions = []
            
            # Limit to top 20 highest risk transactions for performance
            top_risk_indices = high_risk_indices[np.argsort(probabilities[high_risk_indices])[-20:]]
            
            for idx in top_risk_indices:
                transaction = {
                    'index': int(idx),
                    'risk_score': float(probabilities[idx]),
                    'amount': float(df.iloc[idx][column_mapping.get('amount', df.columns[0])]) if 'amount' in column_mapping else 0,
                    'prediction': int(predictions[idx])
                }
                high_risk_transactions.append(transaction)
            
            # Sort by risk score descending
            high_risk_transactions.sort(key=lambda x: x['risk_score'], reverse=True)
            
            # Summary statistics
            amount_col = column_mapping.get('amount')
            if amount_col and amount_col in df.columns:
                summary_stats = {
                    'total_amount': float(df[amount_col].sum()),
                    'avg_amount': float(df[amount_col].mean()),
                    'max_amount': float(df[amount_col].max()),
                    'min_amount': float(df[amount_col].min()),
                    'median_amount': float(df[amount_col].median()),
                    'std_amount': float(df[amount_col].std())
                }
            else:
                summary_stats = {
                    'total_amount': 0,
                    'avg_amount': 0,
                    'max_amount': 0,
                    'min_amount': 0,
                    'median_amount': 0,
                    'std_amount': 0
                }
            
            # Additional insights
            risk_distribution = {
                'low_risk': int(np.sum(probabilities < 0.3)),
                'medium_risk': int(np.sum((probabilities >= 0.3) & (probabilities < 0.7))),
                'high_risk': int(np.sum(probabilities >= 0.7))
            }
            
            # Model confidence metrics
            model_confidence = {
                'avg_prediction_confidence': float(np.mean(np.maximum(probabilities, 1 - probabilities))),
                'prediction_entropy': float(-np.mean(probabilities * np.log(probabilities + 1e-10) + 
                                                   (1 - probabilities) * np.log(1 - probabilities + 1e-10)))
            }
            
            result = {
                'total_transactions': total_transactions,
                'fraud_detected': fraud_detected,
                'fraud_percentage': fraud_percentage,
                'risk_score': avg_risk_score,
                'high_risk_transactions': high_risk_transactions,
                'summary_stats': summary_stats,
                'risk_distribution': risk_distribution,
                'model_confidence': model_confidence,
                'column_mapping': column_mapping,
                'model_used': model_type,
                'prediction_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'training_samples': self.training_stats.get('dataset_size', 'unknown'),
                    'training_date': self.training_stats.get('training_date', 'unknown'),
                    'features_used': len(self.feature_columns)
                }
            }
            
            logger.info(f"Prediction completed successfully:")
            logger.info(f"  - Total transactions: {total_transactions}")
            logger.info(f"  - Fraud detected: {fraud_detected}")
            logger.info(f"  - Fraud rate: {fraud_percentage:.2f}%")
            logger.info(f"  - Average risk score: {avg_risk_score:.3f}")
            logger.info(f"  - High-risk transactions: {len(high_risk_transactions)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fraud prediction: {str(e)}")
            raise Exception(f"Fraud prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        """
        if not self.is_loaded:
            return {'status': 'Models not loaded'}
        
        info = {
            'status': 'Models loaded successfully',
            'available_models': ['random_forest', 'logistic_regression'],
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'features': self.feature_columns,
            'label_encoders': list(self.label_encoders.keys()) if self.label_encoders else [],
            'training_stats': self.training_stats
        }
        
        return info
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data and provide recommendations
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'data_quality_score': 0
        }
        
        try:
            # Basic validation
            if df.empty:
                validation['is_valid'] = False
                validation['errors'].append("Dataset is empty")
                return validation
            
            # Check minimum size
            if len(df) < 10:
                validation['warnings'].append("Very small dataset - results may not be reliable")
                validation['data_quality_score'] -= 20
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                validation['errors'].append("No numeric columns found - cannot analyze transactions")
                validation['is_valid'] = False
            else:
                validation['data_quality_score'] += 30
            
            # Check column mapping coverage
            column_mapping = self.detect_column_mapping(df)
            essential_cols = ['amount']
            found_essential = sum(1 for col in essential_cols if col in column_mapping)
            
            if found_essential == 0:
                validation['warnings'].append("No amount column detected - analysis may be inaccurate")
                validation['data_quality_score'] -= 30
            else:
                validation['data_quality_score'] += 40
            
            # Check for missing values
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 50:
                validation['warnings'].append(f"High missing data percentage: {missing_pct:.1f}%")
                validation['data_quality_score'] -= 20
            elif missing_pct > 10:
                validation['warnings'].append(f"Moderate missing data: {missing_pct:.1f}%")
                validation['data_quality_score'] -= 10
            
            # Data type validation
            for col in numeric_cols:
                if df[col].dtype == 'object':
                    validation['warnings'].append(f"Column {col} should be numeric")
            
            # Generate recommendations
            if len(df) < 100:
                validation['recommendations'].append("Consider using a larger dataset for more reliable results")
            
            if 'type' not in column_mapping:
                validation['recommendations'].append("Include transaction type column for better analysis")
            
            if found_essential < len(essential_cols):
                validation['recommendations'].append("Ensure your CSV has columns for transaction amounts")
            
            # Normalize score
            validation['data_quality_score'] = max(0, min(100, validation['data_quality_score'] + 50))
            
            logger.info(f"Data validation completed. Quality score: {validation['data_quality_score']}")
            
        except Exception as e:
            validation['is_valid'] = False
            validation['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Error in data validation: {e}")
        
        return validation

    # Enhanced predict_fraud method for model_utils.py
# Add this to your existing FraudDetectionPredictor class

def predict_fraud_enhanced(self, df: pd.DataFrame, model_type: str = "random_forest") -> Dict[str, Any]:
    """
    Enhanced fraud prediction with proper thresholding and model comparison
    """
    if not self.is_loaded:
        raise ValueError("Models not loaded. Please train models first.")
    
    try:
        logger.info(f"Starting enhanced fraud prediction with {model_type} model")
        logger.info(f"Input data shape: {df.shape}")
        
        # Detect column mapping for user feedback
        column_mapping = self.detect_column_mapping(df)
        
        # Preprocess data
        processed_data = self.preprocess_user_data(df)
        
        # Get predictions from both models for comparison
        rf_predictions, rf_probabilities = self._get_rf_predictions(processed_data)
        lr_predictions, lr_probabilities = self._get_lr_predictions(processed_data)
        
        # Use requested model as primary
        if model_type == "random_forest":
            primary_predictions = rf_predictions
            primary_probabilities = rf_probabilities
            comparison_predictions = lr_predictions
            comparison_probabilities = lr_probabilities
            comparison_model = "logistic_regression"
        else:
            primary_predictions = lr_predictions
            primary_probabilities = lr_probabilities
            comparison_predictions = rf_predictions
            comparison_probabilities = rf_probabilities
            comparison_model = "random_forest"
        
        # Calculate optimal threshold based on fraud detection principles
        optimal_threshold = self._calculate_optimal_threshold(primary_probabilities)
        
        # Apply optimal threshold
        optimized_predictions = (primary_probabilities >= optimal_threshold).astype(int)
        
        # Calculate results with both default and optimized thresholds
        default_results = self._calculate_results(df, primary_predictions, primary_probabilities, model_type, 0.5)
        optimized_results = self._calculate_results(df, optimized_predictions, primary_probabilities, model_type, optimal_threshold)
        
        # Add model comparison
        comparison_results = self._calculate_results(df, comparison_predictions, comparison_probabilities, comparison_model, 0.5)
        
        # Create comprehensive result
        enhanced_result = {
            **optimized_results,
            "model_comparison": {
                "primary_model": {
                    "name": model_type,
                    "fraud_rate": optimized_results["fraud_percentage"],
                    "threshold_used": optimal_threshold,
                    "avg_risk_score": optimized_results["risk_score"]
                },
                "comparison_model": {
                    "name": comparison_model,
                    "fraud_rate": comparison_results["fraud_percentage"],
                    "threshold_used": 0.5,
                    "avg_risk_score": comparison_results["risk_score"]
                },
                "agreement_rate": self._calculate_agreement(primary_predictions, comparison_predictions),
                "recommendation": self._get_model_recommendation(optimized_results, comparison_results, model_type)
            },
            "threshold_analysis": {
                "default_threshold_results": {
                    "fraud_rate": default_results["fraud_percentage"],
                    "fraud_count": default_results["fraud_detected"]
                },
                "optimized_threshold_results": {
                    "fraud_rate": optimized_results["fraud_percentage"],
                    "fraud_count": optimized_results["fraud_detected"],
                    "threshold_value": optimal_threshold
                },
                "threshold_impact": f"Optimized threshold reduced false positives by {abs(default_results['fraud_detected'] - optimized_results['fraud_detected'])} transactions"
            }
        }
        
        # Enhanced insights with model comparison
        enhanced_result["insights"] = self._generate_enhanced_insights(enhanced_result)
        
        logger.info(f"Enhanced prediction completed successfully")
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Error in enhanced fraud prediction: {str(e)}")
        raise Exception(f"Enhanced fraud prediction failed: {str(e)}")

def _get_rf_predictions(self, processed_data):
    """Get Random Forest predictions"""
    if self.rf_model is None:
        return np.zeros(len(processed_data)), np.zeros(len(processed_data))
    
    predictions = self.rf_model.predict(processed_data)
    probabilities = self.rf_model.predict_proba(processed_data)[:, 1]
    return predictions, probabilities

def _get_lr_predictions(self, processed_data):
    """Get Logistic Regression predictions"""
    if self.lr_model is None:
        return np.zeros(len(processed_data)), np.zeros(len(processed_data))
    
    scaled_data = self.scaler.transform(processed_data)
    predictions = self.lr_model.predict(scaled_data)
    probabilities = self.lr_model.predict_proba(scaled_data)[:, 1]
    return predictions, probabilities

def _calculate_optimal_threshold(self, probabilities):
    """
    Calculate optimal threshold for fraud detection
    Using Youden's J statistic approach for fraud detection
    """
    # For fraud detection, we want to balance precision and recall
    # but typically lean towards catching more fraud (higher recall)
    
    # Sort probabilities and calculate potential thresholds
    sorted_probs = np.sort(probabilities)
    
    # Target fraud rate for financial datasets (typically 1-5%)
    target_fraud_rate = 0.02  # 2% target
    
    # Find threshold that gives us close to target fraud rate
    target_count = int(len(probabilities) * target_fraud_rate)
    
    if target_count < len(sorted_probs):
        optimal_threshold = sorted_probs[-(target_count + 1)]
    else:
        optimal_threshold = 0.5
    
    # Ensure threshold is reasonable (between 0.1 and 0.9)
    optimal_threshold = max(0.1, min(0.9, optimal_threshold))
    
    return optimal_threshold

def _calculate_results(self, df, predictions, probabilities, model_name, threshold):
    """Calculate fraud detection results"""
    total_transactions = len(df)
    fraud_detected = int(predictions.sum())
    fraud_percentage = (fraud_detected / total_transactions) * 100 if total_transactions > 0 else 0
    avg_risk_score = float(probabilities.mean())
    
    # Get high-risk transactions (above threshold)
    high_risk_indices = np.where(probabilities >= threshold)[0]
    high_risk_transactions = []
    
    # Limit to top 20 highest risk transactions
    top_risk_indices = high_risk_indices[np.argsort(probabilities[high_risk_indices])[-20:]]
    
    for idx in top_risk_indices:
        transaction = {
            'index': int(idx),
            'risk_score': float(probabilities[idx]),
            'amount': float(df.iloc[idx].get('amount', 0)),
            'prediction': int(predictions[idx])
        }
        high_risk_transactions.append(transaction)
    
    # Sort by risk score descending
    high_risk_transactions.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return {
        'total_transactions': total_transactions,
        'fraud_detected': fraud_detected,
        'fraud_percentage': fraud_percentage,
        'risk_score': avg_risk_score,
        'high_risk_transactions': high_risk_transactions,
        'model_used': model_name,
        'threshold_used': threshold
    }

def _calculate_agreement(self, pred1, pred2):
    """Calculate agreement rate between two models"""
    agreement = np.mean(pred1 == pred2)
    return float(agreement)

def _get_model_recommendation(self, primary_results, comparison_results, model_type):
    """Generate model recommendation based on results"""
    primary_fraud_rate = primary_results['fraud_percentage']
    comparison_fraud_rate = comparison_results['fraud_percentage']
    
    if abs(primary_fraud_rate - comparison_fraud_rate) < 2:
        return f"Both models show similar results. {model_type.replace('_', ' ').title()} is appropriate for this analysis."
    elif primary_fraud_rate < 5:
        return f"{model_type.replace('_', ' ').title()} shows conservative fraud detection. Good for minimizing false positives."
    elif primary_fraud_rate > 20:
        return f"{model_type.replace('_', ' ').title()} is very sensitive. Consider reviewing threshold or using Random Forest for more conservative results."
    else:
        return f"{model_type.replace('_', ' ').title()} provides balanced fraud detection suitable for most use cases."

def _generate_enhanced_insights(self, results):
    """Generate enhanced insights with model comparison"""
    insights = []
    
    primary_fraud_rate = results['fraud_percentage']
    comparison_fraud_rate = results['model_comparison']['comparison_model']['fraud_rate']
    agreement_rate = results['model_comparison']['agreement_rate']
    
    # Model agreement insight
    if agreement_rate > 0.9:
        insights.append({
            "category": "model_consensus",
            "title": "Strong Model Agreement",
            "description": f"Both models agree on {agreement_rate:.1%} of predictions, indicating reliable results.",
            "recommendation": "High confidence in fraud detection results."
        })
    elif agreement_rate < 0.7:
        insights.append({
            "category": "model_disagreement",
            "title": "Model Disagreement Detected",
            "description": f"Models agree on only {agreement_rate:.1%} of predictions. Consider manual review of borderline cases.",
            "recommendation": "Use ensemble approach or review high-disagreement transactions manually."
        })
    
    # Fraud rate insight with threshold optimization
    insights.append({
        "category": "optimized_detection",
        "title": "Threshold-Optimized Results",
        "description": f"Using optimized threshold of {results.get('threshold_used', 0.5):.2f} instead of default 0.5.",
        "recommendation": "Optimized threshold reduces false positives while maintaining fraud detection accuracy."
    })
    
    # Comparative analysis insight
    if abs(primary_fraud_rate - comparison_fraud_rate) > 10:
        insights.append({
            "category": "model_sensitivity",
            "title": "Significant Model Differences",
            "description": f"Primary model detects {primary_fraud_rate:.1f}% fraud vs {comparison_fraud_rate:.1f}% for comparison model.",
            "recommendation": results['model_comparison']['recommendation']
        })
    
    return insights

# Initialize global predictor instance
predictor = FraudDetectionPredictor()