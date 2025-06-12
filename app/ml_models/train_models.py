import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionTrainer:
    def __init__(self, models_dir="app/ml_models/saved_models"):
        self.models_dir = models_dir
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.training_stats = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Initialized FraudDetectionTrainer with models directory: {self.models_dir}")
    
    def load_paysim_data(self, file_path, sample_size=None, random_state=42):
        """
        Load PaySim dataset with memory optimization and optional sampling
        
        Args:
            file_path (str): Path to the PaySim CSV file
            sample_size (int, optional): Number of samples to load for testing. None for full dataset
            random_state (int): Random state for reproducibility
        
        Returns:
            pd.DataFrame: Loaded and preprocessed dataset
        """
        logger.info(f"Loading PaySim dataset from: {file_path}")
        
        try:
            # First, let's check the file size and get basic info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Dataset file size: {file_size:.2f} MB")
            
            # Define optimal data types to reduce memory usage
            # Note: We'll convert categorical columns after loading to avoid fillna issues
            dtype_dict = {
                'step': 'int32',
                'type': 'str',  # Changed from 'category' to avoid fillna issues
                'amount': 'float32',
                'nameOrig': 'str',  # Changed from 'category'
                'oldbalanceOrg': 'float32',
                'newbalanceOrig': 'float32',
                'nameDest': 'str',  # Changed from 'category'
                'oldbalanceDest': 'float32',
                'newbalanceDest': 'float32',
                'isFraud': 'int8',
                'isFlaggedFraud': 'int8'
            }
            
            # Load data in chunks if it's very large, or all at once if manageable
            if file_size > 1000:  # If larger than 1GB
                logger.info("Large file detected. Loading in chunks...")
                chunks = []
                chunk_size = 100000  # 100k rows per chunk
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=dtype_dict):
                    if sample_size and len(chunks) * chunk_size >= sample_size:
                        chunk = chunk.head(sample_size - len(chunks) * chunk_size)
                        chunks.append(chunk)
                        break
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded {len(df)} rows from chunks")
                
            else:
                # Load entire file at once
                df = pd.read_csv(file_path, dtype=dtype_dict)
                logger.info(f"Loaded {len(df)} rows directly")
                
                # Apply sampling if requested
                if sample_size and len(df) > sample_size:
                    # Stratified sampling to maintain fraud ratio
                    df = df.groupby('isFraud', group_keys=False).apply(
                        lambda x: x.sample(
                            n=min(len(x), sample_size // 2) if x.name == 1 
                            else min(len(x), sample_size - sample_size // 2),
                            random_state=random_state
                        )
                    ).reset_index(drop=True)
                    logger.info(f"Sampled dataset to {len(df)} rows")
            
            # Convert string columns to category after loading (more memory efficient)
            categorical_cols = ['type', 'nameOrig', 'nameDest']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            # Basic dataset statistics
            fraud_count = df['isFraud'].sum()
            fraud_rate = (fraud_count / len(df)) * 100
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  - Total transactions: {len(df):,}")
            logger.info(f"  - Fraudulent transactions: {fraud_count:,}")
            logger.info(f"  - Fraud rate: {fraud_rate:.3f}%")
            logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
            
            self.training_stats['dataset_size'] = len(df)
            self.training_stats['fraud_count'] = fraud_count
            self.training_stats['fraud_rate'] = fraud_rate
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def advanced_feature_engineering(self, df):
        """
        Advanced feature engineering specifically for PaySim dataset
        
        Args:
            df (pd.DataFrame): Raw PaySim dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        logger.info("Starting advanced feature engineering...")
        data = df.copy()
        
        # 1. Time-based features
        logger.info("Creating time-based features...")
        data['hour'] = data['step'] % 24
        data['day'] = data['step'] // 24
        data['is_weekend'] = ((data['day'] % 7) >= 5).astype(int)
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
        data['is_business_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 17)).astype(int)
        
        # 2. Amount-based features
        logger.info("Creating amount-based features...")
        data['amount_log'] = np.log1p(data['amount'])
        data['amount_sqrt'] = np.sqrt(data['amount'])
        
        # Amount percentiles by transaction type
        data['amount_percentile_by_type'] = data.groupby('type')['amount'].rank(pct=True)
        
        # Round amount indicators (potential structuring)
        data['is_round_amount'] = (data['amount'] % 100 == 0).astype(int)
        data['is_very_round_amount'] = (data['amount'] % 1000 == 0).astype(int)
        
        # 3. Balance-based features
        logger.info("Creating balance-based features...")
        # Balance changes
        data['balance_change_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
        data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
        
        # Balance ratios (avoid division by zero)
        data['balance_ratio_orig'] = data['newbalanceOrig'] / (data['oldbalanceOrg'] + 1)
        data['balance_ratio_dest'] = data['newbalanceDest'] / (data['oldbalanceDest'] + 1)
        
        # Balance flags
        data['zeroed_out_orig'] = (data['newbalanceOrig'] == 0).astype(int)
        data['zeroed_out_dest'] = (data['newbalanceDest'] == 0).astype(int)
        data['new_account_orig'] = (data['oldbalanceOrg'] == 0).astype(int)
        data['new_account_dest'] = (data['oldbalanceDest'] == 0).astype(int)
        
        # 4. Transaction-specific features
        logger.info("Creating transaction-specific features...")
        # Encode transaction type
        if 'type' not in self.label_encoders:
            self.label_encoders['type'] = LabelEncoder()
            data['type_encoded'] = self.label_encoders['type'].fit_transform(data['type'])
        else:
            data['type_encoded'] = self.label_encoders['type'].transform(data['type'])
        
        # Customer type inference (basic pattern detection)
        data['orig_is_merchant'] = data['nameOrig'].str.startswith('M').astype(int)
        data['dest_is_merchant'] = data['nameDest'].str.startswith('M').astype(int)
        
        # Transaction amount vs balance ratios
        data['amount_to_orig_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)
        data['amount_to_dest_balance_ratio'] = data['amount'] / (data['oldbalanceDest'] + 1)
        
        # 5. Fraud-specific indicators
        logger.info("Creating fraud-specific indicators...")
        # Exact balance transfer (suspicious pattern)
        data['exact_balance_transfer'] = (
            np.abs(data['balance_change_orig'] + data['balance_change_dest']) < 0.01
        ).astype(int)
        
        # High-risk patterns
        data['high_risk_pattern'] = (
            (data['zeroed_out_orig'] == 1) & 
            (data['amount_to_orig_balance_ratio'] > 0.9)
        ).astype(int)
        
        # Unusual transaction timing
        data['unusual_timing'] = (
            (data['is_night'] == 1) & (data['amount'] > data['amount'].quantile(0.75))
        ).astype(int)
        
        # 6. Aggregated features (memory efficient approach)
        logger.info("Creating aggregated features...")
        # Transaction frequency features (simplified to avoid memory issues)
        type_counts = data['type'].value_counts()
        data['type_frequency'] = data['type'].map(type_counts)
        
        logger.info(f"Feature engineering completed. Dataset shape: {data.shape}")
        return data
    
    def prepare_features(self, df):
        """
        Prepare final feature set for training
        
        Args:
            df (pd.DataFrame): Dataset with engineered features
            
        Returns:
            tuple: (X, y, feature_columns)
        """
        logger.info("Preparing features for training...")
        
        # Select feature columns (excluding target and identifier columns)
        feature_columns = [
            'amount', 'amount_log', 'amount_sqrt', 'amount_percentile_by_type',
            'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_change_orig', 'balance_change_dest', 
            'balance_ratio_orig', 'balance_ratio_dest',
            'zeroed_out_orig', 'zeroed_out_dest', 'new_account_orig', 'new_account_dest',
            'exact_balance_transfer', 'type_encoded', 'orig_is_merchant', 'dest_is_merchant',
            'hour', 'day', 'is_weekend', 'is_night', 'is_business_hours',
            'amount_to_orig_balance_ratio', 'amount_to_dest_balance_ratio',
            'high_risk_pattern', 'is_round_amount', 'is_very_round_amount',
            'unusual_timing', 'type_frequency'
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = set(feature_columns) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        logger.info(f"Using {len(available_features)} features for training")
        
        # Prepare feature matrix - handle categorical columns properly
        X = pd.DataFrame()
        for col in available_features:
            if col in df.columns:
                # Convert categorical columns to numeric if needed
                if df[col].dtype.name == 'category':
                    X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    X[col] = df[col].fillna(0)
            else:
                X[col] = 0
        
        # Ensure all columns are numeric
        X = X.astype('float32')
        
        y = df['isFraud']
        
        # Store feature columns for later use
        self.feature_columns = available_features
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train both Random Forest and Logistic Regression models with optimized parameters
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            test_size (float): Test set proportion
            random_state (int): Random state for reproducibility
            
        Returns:
            dict: Training results and metrics
        """
        logger.info("Starting model training...")
        
        # Split data with stratification to maintain fraud ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Training fraud rate: {(y_train.sum() / len(y_train)) * 100:.3f}%")
        
        # Scale features for Logistic Regression
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # 1. Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,  # Reduced for faster training on large dataset
            max_depth=15,      # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,         # Use all available cores
            class_weight='balanced',  # Handle class imbalance
            max_features='sqrt'       # Reduce overfitting
        )
        
        rf_start_time = datetime.now()
        self.rf_model.fit(X_train, y_train)
        rf_train_time = (datetime.now() - rf_start_time).total_seconds()
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(X_test)
        rf_prob = self.rf_model.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = self.evaluate_model(
            'Random Forest', y_test, rf_pred, rf_prob, rf_train_time
        )
        
        # 2. Train Logistic Regression
        logger.info("Training Logistic Regression...")
        self.lr_model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear',       # Good for small datasets
            penalty='l2'              # L2 regularization
        )
        
        lr_start_time = datetime.now()
        self.lr_model.fit(X_train_scaled, y_train)
        lr_train_time = (datetime.now() - lr_start_time).total_seconds()
        
        # Logistic Regression predictions
        lr_pred = self.lr_model.predict(X_test_scaled)
        lr_prob = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        results['logistic_regression'] = self.evaluate_model(
            'Logistic Regression', y_test, lr_pred, lr_prob, lr_train_time
        )
        
        # Store training metadata
        self.training_stats.update({
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train.shape[1],
            'training_date': datetime.now().isoformat()
        })
        
        return results
    
    def evaluate_model(self, model_name, y_true, y_pred, y_prob, train_time):
        """
        Comprehensive model evaluation
        
        Args:
            model_name (str): Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            train_time (float): Training time in seconds
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Calculate metrics
        auc_score = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        results = {
            'model_name': model_name,
            'auc_score': auc_score,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'npv': npv,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'training_time_seconds': train_time
        }
        
        # Log results
        logger.info(f"{model_name} Results:")
        logger.info(f"  AUC Score: {auc_score:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"  Training Time: {train_time:.2f} seconds")
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
        
        return results
    
    def save_models(self):
        """
        Save trained models and preprocessors with metadata
        """
        logger.info(f"Saving models to {self.models_dir}...")
        
        try:
            # Save models
            joblib.dump(self.rf_model, f"{self.models_dir}/random_forest_model.pkl")
            joblib.dump(self.lr_model, f"{self.models_dir}/logistic_regression_model.pkl")
            
            # Save preprocessors
            joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
            joblib.dump(self.label_encoders, f"{self.models_dir}/label_encoders.pkl")
            joblib.dump(self.feature_columns, f"{self.models_dir}/feature_columns.pkl")
            
            # Save training statistics
            joblib.dump(self.training_stats, f"{self.models_dir}/training_stats.pkl")
            
            logger.info("Models saved successfully!")
            
            # Log saved files
            saved_files = os.listdir(self.models_dir)
            logger.info(f"Saved files: {saved_files}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def create_feature_importance_plot(self, top_n=20):
        """
        Create and save feature importance plot for Random Forest
        
        Args:
            top_n (int): Number of top features to display
        """
        if self.rf_model is None or self.feature_columns is None:
            logger.warning("Models not trained yet. Cannot create feature importance plot.")
            return
        
        logger.info("Creating feature importance plot...")
        
        try:
            # Get feature importance
            importance = self.rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(top_n)
            
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importance (Random Forest)')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.models_dir}/feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to: {plot_path}")
            
            # Log top features
            logger.info("Top 10 most important features:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")

def main():
    """
    Main training pipeline
    """
    logger.info("Starting fraud detection model training pipeline...")
    
    # Configuration
    DATA_PATH = "data/paysim1.csv"  # Update this path to your PaySim dataset
    SAMPLE_SIZE = None  # Use None for full dataset, or specify a number for testing
    RANDOM_STATE = 42
    
    try:
        # Initialize trainer
        trainer = FraudDetectionTrainer()
        
        # Load data
        df = trainer.load_paysim_data(DATA_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
        
        # Feature engineering
        df_engineered = trainer.advanced_feature_engineering(df)
        
        # Prepare features
        X, y, feature_columns = trainer.prepare_features(df_engineered)
        
        # Train models
        results = trainer.train_models(X, y, random_state=RANDOM_STATE)
        
        # Save models
        trainer.save_models()
        
        # Create visualizations
        trainer.create_feature_importance_plot()
        
        # Summary
        logger.info("Training pipeline completed successfully!")
        logger.info("Model comparison:")
        for model_name, metrics in results.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    AUC: {metrics['auc_score']:.4f}")
            logger.info(f"    F1: {metrics['f1_score']:.4f}")
            logger.info(f"    Training time: {metrics['training_time_seconds']:.2f}s")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()