# app/utils/unified_processor.py
"""
Unified data processing module combining all detection and preprocessing logic
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)

class ColumnType(Enum):
    """Column types for fraud detection"""
    AMOUNT = "amount"
    TIMESTAMP = "timestamp" 
    TRANSACTION_TYPE = "transaction_type"
    ACCOUNT_FROM = "account_from"
    ACCOUNT_TO = "account_to"
    BALANCE_BEFORE = "balance_before"
    BALANCE_AFTER = "balance_after"
    FRAUD_LABEL = "fraud_label"
    OTHER = "other"

@dataclass
class DetectionResult:
    """Result of column detection"""
    column_name: str
    detected_type: ColumnType
    confidence: float
    reasons: List[str]
    sample_values: List[Any]

class UnifiedProcessor:
    """
    Unified processor for all data detection, mapping, and preprocessing
    Combines functionality from smart_detector, file_processor, and feature_engineering
    """
    
    def __init__(self):
        self.patterns = self._init_patterns()
        self.paysim_columns = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
        ]
    
    def _init_patterns(self) -> Dict[ColumnType, Dict[str, Any]]:
        """Initialize detection patterns"""
        return {
            ColumnType.AMOUNT: {
                "keywords": ["amount", "value", "sum", "total", "price", "cost"],
                "validator": self._is_amount_column
            },
            ColumnType.TIMESTAMP: {
                "keywords": ["time", "date", "timestamp", "step", "hour"],
                "validator": self._is_timestamp_column
            },
            ColumnType.TRANSACTION_TYPE: {
                "keywords": ["type", "category", "method", "operation"],
                "validator": self._is_transaction_type_column
            },
            ColumnType.ACCOUNT_FROM: {
                "keywords": ["from", "orig", "sender", "nameorig"],
                "validator": self._is_account_column
            },
            ColumnType.ACCOUNT_TO: {
                "keywords": ["to", "dest", "receiver", "namedest"],
                "validator": self._is_account_column
            },
            ColumnType.BALANCE_BEFORE: {
                "keywords": ["oldbalance", "balance_before", "old_balance"],
                "validator": self._is_balance_column
            },
            ColumnType.BALANCE_AFTER: {
                "keywords": ["newbalance", "balance_after", "new_balance"],
                "validator": self._is_balance_column
            },
            ColumnType.FRAUD_LABEL: {
                "keywords": ["fraud", "isfraud", "fraudulent", "label"],
                "validator": self._is_fraud_label_column
            }
        }
    
    def process_file(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main processing function - detects columns, maps them, and converts to PaySim format
        """
        # Step 1: Detect columns
        detection_results = self.detect_columns(df)
        
        # Step 2: Create PaySim mapping
        paysim_mapping = self.create_paysim_mapping(detection_results)
        
        # Step 3: Convert to PaySim format
        paysim_df = self.convert_to_paysim(df, paysim_mapping)
        
        # Step 4: Generate insights
        insights = self.generate_insights(detection_results, df)
        
        return {
            'success': True,
            'detection_results': detection_results,
            'paysim_mapping': paysim_mapping,
            'paysim_data': paysim_df,
            'insights': insights,
            'quality_score': self._calculate_quality_score(detection_results, df)
        }
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, DetectionResult]:
        """Detect column types in DataFrame"""
        results = {}
        
        for col_name in df.columns:
            col_data = df[col_name]
            best_type = ColumnType.OTHER
            best_score = 0.0
            best_reasons = []
            
            # Test each pattern
            for col_type, pattern in self.patterns.items():
                score, reasons = self._calculate_score(col_name, col_data, pattern)
                
                if score > best_score:
                    best_score = score
                    best_type = col_type
                    best_reasons = reasons
            
            # Get sample values
            sample_values = col_data.dropna().head(3).tolist()
            
            results[col_name] = DetectionResult(
                column_name=col_name,
                detected_type=best_type,
                confidence=best_score,
                reasons=best_reasons,
                sample_values=sample_values
            )
        
        return self._resolve_conflicts(results, df)
    
    def _calculate_score(self, col_name: str, col_data: pd.Series, pattern: Dict) -> Tuple[float, List[str]]:
        """Calculate detection score for column"""
        score = 0.0
        reasons = []
        
        # Keyword matching (60% weight)
        keyword_score = self._check_keywords(col_name, pattern["keywords"])
        if keyword_score > 0:
            score += keyword_score * 0.6
            reasons.append(f"Keyword match: {keyword_score:.2f}")
        
        # Content validation (40% weight)
        if "validator" in pattern:
            content_score = pattern["validator"](col_data)
            if content_score > 0:
                score += content_score * 0.4
                reasons.append(f"Content validation: {content_score:.2f}")
        
        return score, reasons
    
    def _check_keywords(self, col_name: str, keywords: List[str]) -> float:
        """Check keyword matches"""
        col_lower = col_name.lower().strip()
        for keyword in keywords:
            if keyword == col_lower:
                return 1.0  # Exact match
            elif keyword in col_lower:
                return 0.8  # Partial match
        return 0.0
    
    # Validator methods
    def _is_amount_column(self, col_data: pd.Series) -> float:
        """Validate amount column"""
        if not pd.api.types.is_numeric_dtype(col_data):
            return 0.0
        
        numeric_data = col_data.dropna()
        if len(numeric_data) == 0:
            return 0.0
        
        score = 0.0
        # Positive values
        if (numeric_data > 0).sum() / len(numeric_data) > 0.8:
            score += 0.5
        # Has variation
        if numeric_data.std() > 0:
            score += 0.3
        # Reasonable range
        if numeric_data.max() / numeric_data.min() > 1.5:
            score += 0.2
        
        return score
    
    def _is_timestamp_column(self, col_data: pd.Series) -> float:
        """Validate timestamp column"""
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return 1.0
        
        # Try parsing as timestamps
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return 0.0
        
        parse_success = 0
        for val in non_null.head(10):
            try:
                pd.to_datetime(str(val))
                parse_success += 1
            except:
                continue
        
        return parse_success / min(10, len(non_null))
    
    def _is_transaction_type_column(self, col_data: pd.Series) -> float:
        """Validate transaction type column"""
        if not pd.api.types.is_object_dtype(col_data):
            return 0.0
        
        unique_ratio = col_data.nunique() / len(col_data)
        if unique_ratio > 0.5:  # Too many unique values
            return 0.0
        
        # Check for known transaction types
        known_types = {'payment', 'transfer', 'deposit', 'withdrawal', 'cash_in', 'cash_out'}
        unique_values = set(str(v).lower() for v in col_data.dropna().unique())
        
        if len(known_types & unique_values) > 0:
            return 0.8
        
        return 0.4 if unique_ratio < 0.1 else 0.0
    
    def _is_account_column(self, col_data: pd.Series) -> float:
        """Validate account ID column"""
        if not pd.api.types.is_object_dtype(col_data):
            return 0.0
        
        unique_ratio = col_data.nunique() / len(col_data)
        return 0.8 if unique_ratio > 0.5 else 0.0
    
    def _is_balance_column(self, col_data: pd.Series) -> float:
        """Validate balance column"""
        if not pd.api.types.is_numeric_dtype(col_data):
            return 0.0
        
        numeric_data = col_data.dropna()
        if len(numeric_data) == 0:
            return 0.0
        
        # Balances should be mostly non-negative
        negative_ratio = (numeric_data < 0).sum() / len(numeric_data)
        return 0.7 if negative_ratio < 0.1 else 0.0
    
    def _is_fraud_label_column(self, col_data: pd.Series) -> float:
        """Validate fraud label column"""
        unique_values = set(col_data.dropna().unique())
        
        if len(unique_values) > 2:
            return 0.0
        
        # Check for binary patterns
        binary_patterns = [
            {0, 1}, {'0', '1'}, {True, False}, 
            {'fraud', 'normal'}, {'yes', 'no'}
        ]
        
        for pattern in binary_patterns:
            if unique_values.issubset(pattern):
                return 0.9
        
        return 0.0
    
    def _resolve_conflicts(self, results: Dict[str, DetectionResult], df: pd.DataFrame) -> Dict[str, DetectionResult]:
        """Resolve detection conflicts"""
        # Ensure at least one amount column
        amount_columns = [name for name, result in results.items() 
                         if result.detected_type == ColumnType.AMOUNT]
        
        if not amount_columns:
            # Find best numeric column
            numeric_cols = []
            for name, result in results.items():
                if pd.api.types.is_numeric_dtype(df[name]):
                    score = self._is_amount_column(df[name])
                    numeric_cols.append((name, score))
            
            if numeric_cols:
                best_col, best_score = max(numeric_cols, key=lambda x: x[1])
                if best_score > 0.2:
                    results[best_col].detected_type = ColumnType.AMOUNT
                    results[best_col].confidence = best_score
                    results[best_col].reasons.append("Auto-assigned as amount")
        
        return results
    
    def create_paysim_mapping(self, detection_results: Dict[str, DetectionResult]) -> Dict[str, str]:
        """Create PaySim column mapping"""
        type_to_paysim = {
            ColumnType.AMOUNT: 'amount',
            ColumnType.TRANSACTION_TYPE: 'type',
            ColumnType.TIMESTAMP: 'step',
            ColumnType.ACCOUNT_FROM: 'nameOrig',
            ColumnType.ACCOUNT_TO: 'nameDest',
            ColumnType.BALANCE_BEFORE: 'oldbalanceOrg',
            ColumnType.BALANCE_AFTER: 'newbalanceOrig',
            ColumnType.FRAUD_LABEL: 'isFraud'
        }
        
        mapping = {}
        for col_name, result in detection_results.items():
            if result.detected_type in type_to_paysim and result.confidence > 0.5:
                paysim_col = type_to_paysim[result.detected_type]
                mapping[paysim_col] = col_name
        
        return mapping
    
    def convert_to_paysim(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Convert DataFrame to PaySim format"""
        paysim_df = pd.DataFrame(index=range(len(df)))
        
        # Set defaults
        defaults = {
            'step': range(1, len(df) + 1),
            'type': 'PAYMENT',
            'amount': 0.0,
            'nameOrig': [f'C{i:08d}' for i in range(len(df))],
            'oldbalanceOrg': 0.0,
            'newbalanceOrig': 0.0,
            'nameDest': [f'M{i%1000:06d}' for i in range(len(df))],
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        
        # Apply mappings
        for paysim_col, source_col in mapping.items():
            if source_col in df.columns:
                if paysim_col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
                    paysim_df[paysim_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(0.0)
                elif paysim_col == 'step':
                    if pd.api.types.is_datetime64_any_dtype(df[source_col]):
                        min_time = df[source_col].min()
                        paysim_df[paysim_col] = ((df[source_col] - min_time).dt.total_seconds() / 3600).astype(int)
                    else:
                        paysim_df[paysim_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(range(1, len(df) + 1))
                else:
                    paysim_df[paysim_col] = df[source_col].astype(str)
        
        # Fill missing columns with defaults
        for col, default in defaults.items():
            if col not in paysim_df.columns:
                paysim_df[col] = default
        
        # Ensure correct column order
        column_order = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                       'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        paysim_df = paysim_df.reindex(columns=column_order, fill_value=0)
        
        return paysim_df
    
    def generate_insights(self, detection_results: Dict[str, DetectionResult], df: pd.DataFrame) -> List[str]:
        """Generate insights about the data"""
        insights = []
        
        # Detection quality
        mapped_cols = sum(1 for r in detection_results.values() if r.detected_type != ColumnType.OTHER)
        total_cols = len(detection_results)
        mapping_rate = mapped_cols / total_cols
        
        if mapping_rate > 0.8:
            insights.append(f"✅ Excellent column detection: {mapped_cols}/{total_cols} columns mapped")
        elif mapping_rate > 0.5:
            insights.append(f"⚠️ Good column detection: {mapped_cols}/{total_cols} columns mapped")
        else:
            insights.append(f"❌ Limited column detection: {mapped_cols}/{total_cols} columns mapped")
        
        # Data quality
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio < 0.05:
            insights.append("✅ High data quality: minimal missing values")
        elif missing_ratio < 0.2:
            insights.append("⚠️ Good data quality: some missing values")
        else:
            insights.append("❌ Data quality issues: significant missing values")
        
        # Analysis readiness
        has_amount = any(r.detected_type == ColumnType.AMOUNT for r in detection_results.values())
        if has_amount:
            insights.append("✅ Ready for fraud analysis: amount column detected")
        else:
            insights.append("❌ Manual mapping required: no amount column detected")
        
        return insights
    
    def _calculate_quality_score(self, detection_results: Dict[str, DetectionResult], df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        score = 50.0  # Base score
        
        # Detection quality (30 points)
        mapped_ratio = sum(1 for r in detection_results.values() if r.detected_type != ColumnType.OTHER) / len(detection_results)
        score += mapped_ratio * 30
        
        # Data completeness (20 points)
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score += (1 - missing_ratio) * 20
        
        return min(100.0, max(0.0, score))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect column types and map them to standard names
        """
        column_mapping = {}
        
        # Amount detection
        amount_keywords = ['amount', 'value', 'sum', 'total', 'price', 'cost']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                if df[col].dtype in ['int64', 'float64']:
                    column_mapping['amount'] = col
                    break
        
        # Transaction type detection
        type_keywords = ['type', 'category', 'kind', 'class', 'method']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in type_keywords):
                if df[col].dtype == 'object':
                    column_mapping['type'] = col
                    break
        
        # Balance detection
        balance_keywords = ['balance', 'bal', 'account']
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in balance_keywords):
                if 'orig' in col_lower or 'sender' in col_lower or 'from' in col_lower:
                    if 'old' in col_lower or 'before' in col_lower or 'prev' in col_lower:
                        column_mapping['oldbalanceOrig'] = col
                    elif 'new' in col_lower or 'after' in col_lower or 'current' in col_lower:
                        column_mapping['newbalanceOrig'] = col
                elif 'dest' in col_lower or 'receiver' in col_lower or 'to' in col_lower:
                    if 'old' in col_lower or 'before' in col_lower or 'prev' in col_lower:
                        column_mapping['oldbalanceDest'] = col
                    elif 'new' in col_lower or 'after' in col_lower or 'current' in col_lower:
                        column_mapping['newbalanceDest'] = col
        
        # Time/Step detection
        time_keywords = ['time', 'timestamp', 'date', 'step', 'hour', 'day']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                column_mapping['step'] = col
                break
        
        # Customer ID detection
        id_keywords = ['id', 'customer', 'account', 'user']
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in id_keywords):
                if 'orig' in col_lower or 'sender' in col_lower or 'from' in col_lower:
                    column_mapping['nameOrig'] = col
                elif 'dest' in col_lower or 'receiver' in col_lower or 'to' in col_lower:
                    column_mapping['nameDest'] = col
        
        return column_mapping
    
    def create_time_features(self, df: pd.DataFrame, time_col: str = None) -> pd.DataFrame:
        """
        Create time-based features
        """
        data = df.copy()
        
        if time_col and time_col in data.columns:
            # If we have a time column, extract features from it
            if data[time_col].dtype == 'object':
                try:
                    data[time_col] = pd.to_datetime(data[time_col])
                except:
                    logger.warning(f"Could not convert {time_col} to datetime")
            
            if pd.api.types.is_datetime64_any_dtype(data[time_col]):
                data['hour'] = data[time_col].dt.hour
                data['day_of_week'] = data[time_col].dt.dayofweek
                data['day_of_month'] = data[time_col].dt.day
                data['month'] = data[time_col].dt.month
            elif pd.api.types.is_numeric_dtype(data[time_col]):
                # Assume it's step-based time (like PaySim)
                data['hour'] = data[time_col] % 24
                data['day'] = data[time_col] // 24
        else:
            # Default time features
            data['hour'] = 12  # Default noon
            data['day'] = 1    # Default day 1
            data['day_of_week'] = 1  # Default Monday
        
        return data
    
    def create_amount_features(self, df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
        """
        Create amount-based features
        """
        data = df.copy()
        
        if amount_col in data.columns:
            amount_series = data[amount_col]
            
            # Basic amount features
            data['amount_log'] = np.log1p(amount_series)
            data['amount_sqrt'] = np.sqrt(amount_series)
            
            # Amount percentiles
            data['amount_percentile'] = amount_series.rank(pct=True)
            
            # Amount z-score
            data['amount_zscore'] = (amount_series - amount_series.mean()) / (amount_series.std() + 1e-8)
            
            # Amount categories
            amount_q25 = amount_series.quantile(0.25)
            amount_q75 = amount_series.quantile(0.75)
            
            data['amount_category'] = pd.cut(amount_series, 
                                           bins=[-np.inf, amount_q25, amount_q75, np.inf],
                                           labels=['low', 'medium', 'high'])
            
            # Round number indicator (possible structuring)
            data['is_round_amount'] = (amount_series % 100 == 0).astype(int)
            data['is_very_round_amount'] = (amount_series % 1000 == 0).astype(int)
        
        return data
    
    def create_balance_features(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Create balance-based features
        """
        data = df.copy()
        
        # Default values
        data['oldbalanceOrig'] = data.get(column_mapping.get('oldbalanceOrig', 'oldbalanceOrig'), 0)
        data['newbalanceOrig'] = data.get(column_mapping.get('newbalanceOrig', 'newbalanceOrig'), 0)
        data['oldbalanceDest'] = data.get(column_mapping.get('oldbalanceDest', 'oldbalanceDest'), 0)
        data['newbalanceDest'] = data.get(column_mapping.get('newbalanceDest', 'newbalanceDest'), 0)
        
        # Balance change features
        data['balance_change_orig'] = data['newbalanceOrig'] - data['oldbalanceOrig']
        data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
        
        # Balance ratios (avoid division by zero)
        data['balance_ratio_orig'] = data['newbalanceOrig'] / (data['oldbalanceOrig'] + 1)
        data['balance_ratio_dest'] = data['newbalanceDest'] / (data['oldbalanceDest'] + 1)
        
        # Balance flags
        data['zeroed_out_orig'] = (data['newbalanceOrig'] == 0).astype(int)
        data['zeroed_out_dest'] = (data['newbalanceDest'] == 0).astype(int)
        data['new_account_orig'] = (data['oldbalanceOrig'] == 0).astype(int)
        data['new_account_dest'] = (data['oldbalanceDest'] == 0).astype(int)
        
        # Suspicious balance patterns
        data['exact_balance_transfer'] = (
            np.abs(data['balance_change_orig'] + data['balance_change_dest']) < 0.01
        ).astype(int)
        
        return data
    
    def create_transaction_features(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Create transaction-specific features
        """
        data = df.copy()
        
        # Transaction type encoding
        type_col = column_mapping.get('type', 'type')
        if type_col in data.columns:
            if type_col not in self.label_encoders:
                self.label_encoders[type_col] = LabelEncoder()
                data['type_encoded'] = self.label_encoders[type_col].fit_transform(data[type_col].fillna('UNKNOWN'))
            else:
                # Handle unseen categories
                known_categories = self.label_encoders[type_col].classes_
                data[type_col] = data[type_col].fillna('UNKNOWN')
                data[type_col] = data[type_col].apply(
                    lambda x: x if x in known_categories else 'UNKNOWN'
                )
                data['type_encoded'] = self.label_encoders[type_col].transform(data[type_col])
        else:
            data['type_encoded'] = 0  # Default encoding
        
        # Customer ID features (if available)
        orig_col = column_mapping.get('nameOrig')
        dest_col = column_mapping.get('nameDest')
        
        if orig_col and dest_col:
            # Self-transfer detection
            data['is_self_transfer'] = (data[orig_col] == data[dest_col]).astype(int)
            
            # Customer type inference (basic pattern detection)
            data['orig_is_merchant'] = data[orig_col].str.startswith('M').astype(int)
            data['dest_is_merchant'] = data[dest_col].str.startswith('M').astype(int)
        else:
            data['is_self_transfer'] = 0
            data['orig_is_merchant'] = 0
            data['dest_is_merchant'] = 0
        
        return data
    
    def create_derived_features(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Create advanced derived features
        """
        data = df.copy()
        
        # Amount vs balance ratios
        amount_col = column_mapping.get('amount', 'amount')
        if amount_col in data.columns:
            data['amount_to_orig_balance_ratio'] = data[amount_col] / (data['oldbalanceOrig'] + 1)
            data['amount_to_dest_balance_ratio'] = data[amount_col] / (data['oldbalanceDest'] + 1)
            
            # Large transaction flags
            amount_mean = data[amount_col].mean()
            amount_std = data[amount_col].std()
            data['is_large_transaction'] = (
                data[amount_col] > amount_mean + 2 * amount_std
            ).astype(int)
            
            # Velocity features (if we have multiple transactions)
            if len(data) > 1:
                data['amount_rank'] = data[amount_col].rank(pct=True)
        
        # Risk indicators
        data['high_risk_pattern'] = (
            (data['zeroed_out_orig'] == 1) & 
            (data['amount_to_orig_balance_ratio'] > 0.9)
        ).astype(int)
        
        return data
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit the feature engineer and transform the data
        """
        # Detect column mapping
        column_mapping = self.detect_column_types(df)
        logger.info(f"Detected column mapping: {column_mapping}")
        
        # Create features step by step
        data = df.copy()
        
        # Time features
        data = self.create_time_features(data, column_mapping.get('step'))
        
        # Amount features
        amount_col = column_mapping.get('amount', 'amount')
        if amount_col in data.columns:
            data = self.create_amount_features(data, amount_col)
        
        # Balance features
        data = self.create_balance_features(data, column_mapping)
        
        # Transaction features
        data = self.create_transaction_features(data, column_mapping)
        
        # Derived features
        data = self.create_derived_features(data, column_mapping)
        
        # Select final feature columns
        feature_columns = [
            'amount', 'amount_log', 'amount_sqrt', 'amount_zscore', 'amount_percentile',
            'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_change_orig', 'balance_change_dest', 
            'balance_ratio_orig', 'balance_ratio_dest',
            'zeroed_out_orig', 'zeroed_out_dest', 'new_account_orig', 'new_account_dest',
            'exact_balance_transfer', 'type_encoded', 'is_self_transfer',
            'orig_is_merchant', 'dest_is_merchant', 'hour', 'day',
            'amount_to_orig_balance_ratio', 'amount_to_dest_balance_ratio',
            'is_large_transaction', 'high_risk_pattern', 'is_round_amount'
        ]
        
        # Filter to existing columns and ensure no missing values
        available_features = [col for col in feature_columns if col in data.columns]
        feature_data = data[available_features].fillna(0)
        
        # Fit scaler on numeric features
        numeric_features = feature_data.select_dtypes(include=[np.number]).columns
        feature_data[numeric_features] = self.scaler.fit_transform(feature_data[numeric_features])
        
        self.feature_columns = available_features
        self.is_fitted = True
        
        return feature_data, available_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineer
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Apply same transformations as fit_transform
        column_mapping = self.detect_column_types(df)
        data = df.copy()
        
        # Time features
        data = self.create_time_features(data, column_mapping.get('step'))
        
        # Amount features
        amount_col = column_mapping.get('amount', 'amount')
        if amount_col in data.columns:
            data = self.create_amount_features(data, amount_col)
        
        # Balance features
        data = self.create_balance_features(data, column_mapping)
        
        # Transaction features
        data = self.create_transaction_features(data, column_mapping)
        
        # Derived features
        data = self.create_derived_features(data, column_mapping)
        
        # Select and prepare final features
        feature_data = pd.DataFrame()
        for col in self.feature_columns:
            if col in data.columns:
                feature_data[col] = data[col]
            else:
                feature_data[col] = 0  # Default value for missing features
        
        # Scale numeric features
        numeric_features = feature_data.select_dtypes(include=[np.number]).columns
        feature_data[numeric_features] = self.scaler.transform(feature_data[numeric_features])
        
        return feature_data.fillna(0)


def create_advanced_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Convenience function to create advanced features
    """
    engineer = FeatureEngineer()
    
    if is_training:
        features, feature_cols = engineer.fit_transform(df)
        return features, engineer
    else:
        return engineer.transform(df)
    
logger = logging.getLogger(__name__)

class FileProcessor:
    """
    Enhanced file processor with smart column detection
    """
    
    def __init__(self, max_file_size: int = 500 * 1024 * 1024):  # 500MB default
        self.max_file_size = max_file_size
        self.supported_formats = ['.csv']
        self.detector = UnifiedProcessor()
        
    def process_file_with_smart_detection(self, file_content: bytes, 
                                        filename: str) -> Dict[str, Any]:
        """
        Process uploaded file with smart column detection
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Comprehensive analysis results
        """
        result = {
            'success': False,
            'data': None,
            'file_info': {},
            'column_analysis': {},
            'analysis_preview': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Validate file
            validation = self._validate_file(file_content, filename)
            result['file_info'] = validation['file_info']
            
            if not validation['is_valid']:
                result['errors'].extend(validation['errors'])
                return result
            
            result['warnings'].extend(validation['warnings'])
            
            # Step 2: Read CSV safely
            df, read_errors = self._read_csv_safe(file_content)
            if df is None:
                result['errors'].extend(read_errors)
                return result
            
            logger.info(f"Successfully read CSV: {len(df)} rows, {len(df.columns)} columns")
            
            # Step 3: Smart column detection
            detection_results = self.detector.detect_columns(df)
            
            # Step 4: Analyze detection quality
            quality_analysis = self._analyze_detection_quality(detection_results, df)
            
            # Step 5: Generate recommendations
            recommendations = self.detector.get_analysis_recommendations(detection_results)
            
            # Step 6: Create analysis preview
            analysis_preview = self._create_analysis_preview(detection_results, df)
            
            # Step 7: Determine available analysis types
            available_analyses = self._determine_available_analyses(detection_results)
            
            # Step 8: Create PaySim mapping
            paysim_mapping = self.detector.create_paysim_mapping(detection_results)
            
            # Populate successful result
            result.update({
                'success': True,
                'data': df,
                'column_analysis': {
                    'detection_results': detection_results,
                    'quality_analysis': quality_analysis,
                    'recommendations': recommendations,
                    'paysim_mapping': paysim_mapping,
                    'unmapped_columns': [
                        name for name, res in detection_results.items()
                        if res.detected_type == ColumnType.OTHER
                    ]
                },
                'analysis_preview': analysis_preview,
                'available_analyses': available_analyses
            })
            
            logger.info(f"Smart detection completed for {filename}")
            
        except Exception as e:
            result['errors'].append(f"Processing error: {str(e)}")
            logger.error(f"Error processing {filename}: {str(e)}")
        
        return result
    
    def _validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Validate uploaded file"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        # Check file size
        size_mb = len(file_content) / 1024 / 1024
        if len(file_content) > self.max_file_size:
            validation_result['is_valid'] = False
            validation_result['errors'].append(
                f"File size ({size_mb:.1f}MB) exceeds maximum allowed size "
                f"({self.max_file_size / 1024 / 1024:.0f}MB)"
            )
        
        # Check file extension
        file_path = Path(filename)
        if file_path.suffix.lower() not in self.supported_formats:
            validation_result['is_valid'] = False
            validation_result['errors'].append(
                f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Size warnings
        if size_mb > 100:
            validation_result['warnings'].append(
                "Large file detected. Processing may take longer."
            )
        elif size_mb > 50:
            validation_result['warnings'].append(
                "Medium-large file. Analysis will be optimized for performance."
            )
        
        # Store file info
        validation_result['file_info'] = {
            'filename': filename,
            'size_bytes': len(file_content),
            'size_mb': size_mb,
            'extension': file_path.suffix.lower()
        }
        
        return validation_result
    
    def _read_csv_safe(self, file_content: bytes) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Safely read CSV with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']
        errors = []
        
        for encoding in encodings:
            for separator in separators:
                try:
                    # Decode and read
                    text_content = file_content.decode(encoding)
                    df = pd.read_csv(
                        io.StringIO(text_content),
                        sep=separator,
                        low_memory=False,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a', '#N/A'],
                        keep_default_na=True
                    )
                    
                    # Validate DataFrame
                    if self._validate_dataframe(df):
                        logger.info(f"CSV read successfully: encoding={encoding}, separator='{separator}'")
                        return df, []
                        
                except Exception as e:
                    errors.append(f"Failed with encoding={encoding}, separator='{separator}': {str(e)}")
                    continue
        
        return None, errors
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame is suitable for analysis"""
        if df.empty:
            return False
        
        if len(df.columns) < 2:
            return False
        
        if len(df) < 5:  # Need at least 5 rows
            return False
        
        # Check if we have some real data (not all NaN)
        if df.isnull().all().all():
            return False
        
        # Check for reasonable column names
        unnamed_cols = sum(1 for col in df.columns if 'Unnamed:' in str(col))
        if unnamed_cols > len(df.columns) * 0.7:  # More than 70% unnamed
            return False
        
        return True
    
    def _analyze_detection_quality(self, detection_results: Dict[str, DetectionResult],
                                 df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall quality of column detection"""
        
        confidences = [result.confidence for result in detection_results.values()]
        
        # Count mapped vs unmapped columns
        mapped_count = sum(1 for result in detection_results.values() 
                          if result.detected_type != ColumnType.OTHER)
        total_count = len(detection_results)
        
        # Check for essential columns
        detected_types = {result.detected_type for result in detection_results.values()}
        has_amount = ColumnType.AMOUNT in detected_types
        has_timestamp = ColumnType.TIMESTAMP in detected_types
        has_accounts = (ColumnType.ACCOUNT_FROM in detected_types or 
                       ColumnType.ACCOUNT_TO in detected_types)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine if ready for analysis
        ready_for_analysis = (
            has_amount and 
            overall_confidence > 0.5 and 
            mapped_count >= max(2, total_count * 0.3)  # At least 30% mapped
        )
        
        return {
            "overall_confidence": overall_confidence,
            "mapped_columns_ratio": mapped_count / total_count if total_count > 0 else 0.0,
            "essential_columns": {
                "amount": has_amount,
                "timestamp": has_timestamp, 
                "accounts": has_accounts
            },
            "ready_for_analysis": ready_for_analysis,
            "total_columns": total_count,
            "mapped_columns": mapped_count,
            "data_quality_score": self._calculate_data_quality_score(df)
        }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        score = 100.0
        
        # Missing data penalty
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 30
        
        # Duplicate rows penalty  
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 20
        
        # Very low uniqueness penalty (all same values)
        low_uniqueness_cols = sum(1 for col in df.columns 
                                 if df[col].nunique() == 1)
        if low_uniqueness_cols > 0:
            score -= (low_uniqueness_cols / len(df.columns)) * 10
        
        return max(0.0, min(100.0, score))
    
    def _create_analysis_preview(self, detection_results: Dict[str, DetectionResult],
                               df: pd.DataFrame) -> Dict[str, Any]:
        """Create preview of analysis capabilities"""
        preview = {
            "sample_data": {},
            "statistics": {},
            "potential_insights": []
        }
        
        # Sample data for detected columns
        for column_name, result in detection_results.items():
            if result.detected_type != ColumnType.OTHER and result.confidence > 0.6:
                preview["sample_data"][f"{result.detected_type.value} ({column_name})"] = {
                    "sample_values": result.sample_values,
                    "confidence": f"{result.confidence:.1%}",
                    "data_type": str(df[column_name].dtype)
                }
        
        # Basic statistics
        detected_types = {result.detected_type for result in detection_results.values() 
                         if result.confidence > 0.6}
        
        preview["statistics"] = {
            "Total Rows": len(df),
            "Total Columns": len(df.columns),
            "Detected Columns": len([r for r in detection_results.values() 
                                   if r.detected_type != ColumnType.OTHER]),
            "Average Confidence": f"{np.mean([r.confidence for r in detection_results.values()]):.1%}"
        }
        
        # Potential insights based on detected columns
        if ColumnType.AMOUNT in detected_types:
            preview["potential_insights"].append("💰 Transaction amount analysis possible")
        
        if ColumnType.TIMESTAMP in detected_types:
            preview["potential_insights"].append("⏰ Temporal pattern analysis available")
        
        if ColumnType.TRANSACTION_TYPE in detected_types:
            preview["potential_insights"].append("📊 Transaction category analysis possible")
        
        if {ColumnType.ACCOUNT_FROM, ColumnType.ACCOUNT_TO} & detected_types:
            preview["potential_insights"].append("🔗 Network relationship analysis available")
        
        if ColumnType.FRAUD_LABEL in detected_types:
            preview["potential_insights"].append("🎯 Supervised learning with labels possible")
        
        balance_types = {ColumnType.BALANCE_BEFORE, ColumnType.BALANCE_AFTER} & detected_types
        if balance_types:
            preview["potential_insights"].append("📈 Balance change analysis available")
        
        return preview
    
    def _determine_available_analyses(self, detection_results: Dict[str, DetectionResult]) -> List[Dict[str, Any]]:
        """Determine what analysis types are possible"""
        detected_types = {result.detected_type for result in detection_results.values() 
                         if result.confidence > 0.6}
        analyses = []
        
        # Basic fraud detection (needs amount)
        if ColumnType.AMOUNT in detected_types:
            analyses.append({
                "name": "Basic Fraud Detection",
                "description": "Statistical anomaly detection using transaction amounts",
                "confidence": 0.8,
                "requirements_met": ["amount"],
                "optional_available": [t.value for t in detected_types if t != ColumnType.AMOUNT]
            })
        
        # Enhanced fraud detection (needs amount + balance)
        if (ColumnType.AMOUNT in detected_types and 
            ColumnType.BALANCE_BEFORE in detected_types):
            analyses.append({
                "name": "Enhanced Fraud Detection",
                "description": "Advanced detection with balance tracking",
                "confidence": 0.95,
                "requirements_met": ["amount", "balance_before"],
                "optional_available": [t.value for t in detected_types]
            })
        
        # Network analysis (needs amount + accounts)
        if (ColumnType.AMOUNT in detected_types and 
            (ColumnType.ACCOUNT_FROM in detected_types or ColumnType.ACCOUNT_TO in detected_types)):
            analyses.append({
                "name": "Network Analysis", 
                "description": "Transaction network and relationship analysis",
                "confidence": 0.85,
                "requirements_met": ["amount", "accounts"],
                "optional_available": [t.value for t in detected_types]
            })
        
        # Temporal analysis (needs amount + timestamp)
        if (ColumnType.AMOUNT in detected_types and 
            ColumnType.TIMESTAMP in detected_types):
            analyses.append({
                "name": "Temporal Analysis",
                "description": "Time-based pattern and velocity analysis", 
                "confidence": 0.9,
                "requirements_met": ["amount", "timestamp"],
                "optional_available": [t.value for t in detected_types]
            })
        
        # Supervised learning (needs amount + fraud labels)
        if (ColumnType.AMOUNT in detected_types and 
            ColumnType.FRAUD_LABEL in detected_types):
            analyses.append({
                "name": "Supervised Learning",
                "description": "Model training with ground truth labels",
                "confidence": 0.98,
                "requirements_met": ["amount", "fraud_labels"],
                "optional_available": [t.value for t in detected_types]
            })
        
        # Sort by confidence
        analyses.sort(key=lambda x: x["confidence"], reverse=True)
        
        return analyses
    
    def convert_to_paysim_format(self, df: pd.DataFrame, 
                                detection_results: Dict[str, DetectionResult],
                                analysis_type: str = "Basic Fraud Detection") -> pd.DataFrame:
        """
        Convert any CSV format to PaySim-compatible format - COMPLETELY FIXED VERSION
        """
        logger.info(f"Converting to PaySim format for {analysis_type}")
        
        # Get the mapping
        paysim_mapping = self.detector.create_paysim_mapping(detection_results)
        
        # Create empty DataFrame with proper index
        paysim_df = pd.DataFrame(index=range(len(df)))
        
        # Start with sequential/list columns that need special handling
        paysim_df['step'] = range(1, len(df) + 1)
        paysim_df['nameOrig'] = [f'C{i:08d}' for i in range(len(df))]
        paysim_df['nameDest'] = [f'M{i%1000:06d}' for i in range(len(df))]
        
        # Scalar defaults for remaining columns
        scalar_defaults = {
            'type': 'PAYMENT',
            'amount': 0.0,
            'oldbalanceOrg': 0.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        
        # Map detected columns
        for paysim_col, source_col in paysim_mapping.items():
            if source_col in df.columns:
                try:
                    if paysim_col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
                        # Numeric columns
                        paysim_df[paysim_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(0.0)
                    elif paysim_col == 'step':
                        # Time/step column
                        if pd.api.types.is_datetime64_any_dtype(df[source_col]):
                            min_time = df[source_col].min()
                            paysim_df[paysim_col] = ((df[source_col] - min_time).dt.total_seconds() / 3600).astype(int)
                        else:
                            numeric_vals = pd.to_numeric(df[source_col], errors='coerce')
                            if numeric_vals.notna().any():
                                paysim_df[paysim_col] = numeric_vals.fillna(range(1, len(df) + 1))
                            # If conversion fails, keep the default range we already set
                    elif paysim_col == 'type':
                        # Transaction type
                        paysim_df[paysim_col] = self._normalize_transaction_types(df[source_col])
                    elif paysim_col in ['nameOrig', 'nameDest']:
                        # Account columns
                        paysim_df[paysim_col] = df[source_col].astype(str)
                except Exception as e:
                    logger.warning(f"Error mapping {paysim_col} from {source_col}: {e}")
                    # Keep the default value
        
        # Fill any missing columns with defaults
        for paysim_col, default_value in scalar_defaults.items():
            if paysim_col not in paysim_df.columns:
                paysim_df[paysim_col] = default_value
        
        # Ensure correct data types
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in numeric_cols:
            if col in paysim_df.columns:
                paysim_df[col] = pd.to_numeric(paysim_df[col], errors='coerce').fillna(0)
        
        string_cols = ['type', 'nameOrig', 'nameDest']
        for col in string_cols:
            if col in paysim_df.columns:
                paysim_df[col] = paysim_df[col].astype(str)
        
        logger.info(f"PaySim conversion completed: {len(paysim_df)} rows, {len(paysim_df.columns)} columns")
        return paysim_df
    
    def _normalize_transaction_types(self, type_series: pd.Series) -> pd.Series:
        """Normalize transaction types to PaySim format"""
        type_mapping = {
            'payment': 'PAYMENT',
            'pay': 'PAYMENT', 
            'purchase': 'PAYMENT',
            'transfer': 'TRANSFER',
            'send': 'TRANSFER',
            'deposit': 'CASH_IN',
            'cash_in': 'CASH_IN',
            'withdrawal': 'CASH_OUT',
            'cash_out': 'CASH_OUT',
            'withdraw': 'CASH_OUT',
            'debit': 'DEBIT'
        }
        
        normalized = type_series.astype(str).str.lower().map(type_mapping)
        return normalized.fillna('PAYMENT')  # Default to PAYMENT


# Create global instance for easy import
file_processor = FileProcessor()

# Factory function for easy import
def create_processor() -> UnifiedProcessor:
    """Create a new unified processor instance"""
    return UnifiedProcessor()