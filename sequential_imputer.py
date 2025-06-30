# sequential_imputer.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class SequentialImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_impute, initial_features):
        """
        Initializes the SequentialImputer.

        Parameters:
            columns_to_impute (list): List of columns to impute sequentially.
            initial_features (list): List of initial feature columns.
        """
        self.columns_to_impute = columns_to_impute
        self.initial_features = initial_features
        self.models = {}
        self.feature_columns = initial_features.copy()
        
    def _standardize_and_convert_to_str(self, X):
        """
        Standardizes numerical columns and converts all columns to strings.

        Parameters:
            X (DataFrame): Input DataFrame.

        Returns:
            DataFrame: DataFrame with all columns as strings.
        """
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].round(2)
            X[col] = X[col].astype(str)
        return X

    def fit(self, X, y=None):
        X = X.copy()
        self.feature_columns = self.initial_features.copy()
        
        for col in self.columns_to_impute:
            known_mask = X[col] != 'UNKNOWN'
            if known_mask.sum() == 0:
                continue  # Skip if no known values

            # Prepare training data
            X_known = X.loc[known_mask, self.feature_columns]
            y_known = X.loc[known_mask, col]

            # Standardize and convert all features to strings
            X_known = self._standardize_and_convert_to_str(X_known)

            # One-hot encode features with prefixes
            X_known_encoded = pd.get_dummies(
                X_known, drop_first=True, prefix=X_known.columns, prefix_sep='__'
            )

            # Check for duplicates
            if X_known_encoded.columns.duplicated().any():
                duplicates = X_known_encoded.columns[X_known_encoded.columns.duplicated()]
                raise ValueError(f"Duplicate columns detected after encoding: {duplicates}")

            # Train the model
            model = RandomForestClassifier()
            model.fit(X_known_encoded, y_known)
            self.models[col] = model

            # Update feature columns
            self.feature_columns.append(col)
        
        return self

    def transform(self, X):
        X = X.copy()
        feature_columns = self.initial_features.copy()
        
        for col in self.columns_to_impute:
            unknown_mask = X[col] == 'UNKNOWN'
            if unknown_mask.sum() == 0:
                # Update feature columns even if no unknowns to keep sequence
                feature_columns.append(col)
                continue  # Skip if no unknown values

            # Prepare data for prediction
            X_unknown = X.loc[unknown_mask, feature_columns]

            # Standardize and convert all features to strings
            X_unknown = self._standardize_and_convert_to_str(X_unknown)

            # One-hot encode features with prefixes
            X_unknown_encoded = pd.get_dummies(
                X_unknown, drop_first=True, prefix=X_unknown.columns, prefix_sep='__'
            )

            # Align columns with training data
            model = self.models[col]
            model_features = model.feature_names_in_

            # Reindex ensuring there are no duplicates
            X_unknown_encoded = X_unknown_encoded.reindex(columns=model_features, fill_value=0)

            # Predict and impute missing values
            X.loc[unknown_mask, col] = model.predict(X_unknown_encoded)

            # Update feature columns
            feature_columns.append(col)
        
        return X
