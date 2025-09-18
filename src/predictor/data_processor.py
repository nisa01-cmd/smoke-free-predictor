import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from .config import config

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, cleaning, preprocessing, and feature engineering
    for the Smoke-Free Predictor application.
    """

    def __init__(self, config_override: Optional[Dict] = None):
        self.config = config
        if config_override:
            for key, value in config_override.items():
                setattr(self.config.data, key, value)

        self.preprocessor: Optional[ColumnTransformer] = None
        self._target_encoder: Optional[LabelEncoder] = None

    # --------------------- Data Loading & Cleaning --------------------- #

    def load_data(self, file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Loaded data from {file_path}. Shape: {df.shape}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        original_shape = df.shape
        df = df.drop_duplicates()
        if df.shape[0] < original_shape[0]:
            logger.info(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")

        # Optional outlier handling
        if self.config.data.outlier_detection:
            df = self._handle_outliers(df)

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        method = self.config.data.outlier_method.lower()
        if method == "iqr":
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        elif method == "zscore":
            for col in numeric_cols:
                z = (df[col] - df[col].mean()) / df[col].std()
                df = df[np.abs(z) <= 3]
        return df

    # --------------------- Feature Preparation --------------------- #

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Auto-detect column types
        if self.config.data.numeric_features is None:
            self.config.data.numeric_features = list(df.select_dtypes(include=[np.number]).columns)
        if self.config.data.categorical_features is None:
            self.config.data.categorical_features = list(df.select_dtypes(include=['object', 'category']).columns)
        if self.config.data.datetime_features is None:
            self.config.data.datetime_features = list(df.select_dtypes(include=['datetime64']).columns)

        # Remove target & ID from features
        for col in [self.config.data.target_column, self.config.data.id_column]:
            if col:
                for lst in [self.config.data.numeric_features, self.config.data.categorical_features, self.config.data.datetime_features]:
                    if col in lst:
                        lst.remove(col)

        # Engineer datetime features
        for col in self.config.data.datetime_features:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                self.config.data.numeric_features.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])

        return df

    # --------------------- Preprocessor --------------------- #

    def create_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy=self.config.data.missing_value_strategy)),
            ("scaler", StandardScaler() if self.config.model.feature_scaling else 'passthrough')
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.config.data.numeric_features),
            ("cat", categorical_transformer, self.config.data.categorical_features)
        ])
        return self.preprocessor

    # --------------------- Training / Preprocessing --------------------- #

    def load_and_preprocess(self, file_path: str, target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.prepare_features(df)

        target_col = target_column or self.config.data.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = df.drop(columns=[target_col] + ([self.config.data.id_column] if self.config.data.id_column in df.columns else []))
        y = df[target_col]

        preprocessor = self.create_preprocessor()
        X_processed = preprocessor.fit_transform(X)

        # Always use LabelEncoder for classification targets
        self._target_encoder = LabelEncoder()
        y_processed = self._target_encoder.fit_transform(y)

        logger.info(f"Preprocessing completed. X shape: {X_processed.shape}, y shape: {y_processed.shape}")
        return X_processed, y_processed

    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call load_and_preprocess first.")

        df_clean = self.clean_data(df.copy())
        df_prepared = self.prepare_features(df_clean)

        if self.config.data.id_column and self.config.data.id_column in df_prepared.columns:
            df_prepared = df_prepared.drop(columns=[self.config.data.id_column])

        X_transformed = self.preprocessor.transform(df_prepared)
        return X_transformed

    # --------------------- Feature Names --------------------- #

    def get_feature_names(self) -> List[str]:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")

        feature_names = self.config.data.numeric_features.copy()
        cat_encoder = self.preprocessor.named_transformers_['cat']['encoder']
        cat_names = cat_encoder.get_feature_names_out(self.config.data.categorical_features)
        feature_names.extend(cat_names)
        return feature_names

    # --------------------- Save / Load Preprocessor --------------------- #

    def save_preprocessor(self, file_path: str) -> None:
        if self.preprocessor is None:
            raise ValueError("No preprocessor fitted.")
        joblib.dump({
            "preprocessor": self.preprocessor,
            "target_encoder": self._target_encoder,
            "config": self.config.data
        }, file_path)
        logger.info(f"Preprocessor saved to: {file_path}")

    def load_preprocessor(self, file_path: str) -> None:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {file_path}")
        data = joblib.load(file_path)
        self.preprocessor = data["preprocessor"]
        self._target_encoder = data["target_encoder"]
        self.config.data = data["config"]
        logger.info(f"Preprocessor loaded from: {file_path}")
