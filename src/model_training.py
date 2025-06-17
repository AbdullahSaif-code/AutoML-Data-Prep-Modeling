import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error, classification_report,
    roc_auc_score, log_loss
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Enhanced model training class with comprehensive functionality.
    """
    
    def __init__(self, data, target_column, problem_type='auto'):
        """
        Initialize the ModelTrainer.
        
        Args:
            data: pandas DataFrame containing the dataset
            target_column: string name of the target column
            problem_type: 'classification', 'regression', or 'auto' to detect automatically
        """
        self.data = data.copy()
        self.target_column = target_column
        self.original_problem_type = problem_type
        self.problem_type = self._determine_problem_type() if problem_type == 'auto' else problem_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cv_scores = None
        self.best_params = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data and parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")
        
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError("Problem type must be 'classification' or 'regression'.")
        
        # Check for sufficient data
        if len(self.data) < 10:
            raise ValueError("Dataset is too small. Need at least 10 samples.")
    
    def _determine_problem_type(self):
        """Automatically determine if this is a classification or regression problem."""
        target_data = self.data[self.target_column].dropna()
        
        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(target_data):
            return 'classification'
        
        # Check unique values ratio
        unique_ratio = target_data.nunique() / len(target_data)
        
        # If less than 5% unique values or less than 20 unique values, likely classification
        if unique_ratio < 0.05 or target_data.nunique() < 20:
            return 'classification'
        else:
            return 'regression'
    
    def get_available_models(self):
        """Get list of available models for the current problem type."""
        if self.problem_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Naive Bayes': GaussianNB(),
                'K-Nearest Neighbors': KNeighborsClassifier()
            }
        else:  # regression
            return {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'SVR': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42),
                'Elastic Net': ElasticNet(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor()
            }
    
    def split_data(self, feature_columns=None, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into training and testing sets.
        
        Args:
            feature_columns: list of feature column names (None for all except target)
            test_size: float, proportion of test set
            random_state: int, random seed
            stratify: bool, whether to stratify split for classification
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if feature_columns is None:
            feature_columns = [col for col in self.data.columns if col != self.target_column]
        
        # Validate feature columns
        missing_cols = [col for col in feature_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
        
        self.feature_columns = feature_columns
        
        # Prepare features and target
        X = self.data[feature_columns].copy()
        y = self.data[self.target_column].copy()
        
        # Handle missing values in features
        if X.isnull().any().any():
            print("Warning: Missing values detected in features. Consider preprocessing the data.")
        
        # Handle missing values in target
        if y.isnull().any():
            print("Warning: Missing values in target variable will be removed.")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
        
        # Stratification for classification
        stratify_param = None
        if self.problem_type == 'classification' and stratify:
            try:
                stratify_param = y
            except:
                print("Warning: Could not stratify. Proceeding with random split.")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def select_model(self, model_name):
        """
        Select a model by name or pass a model object directly.
        
        Args:
            model_name: string name of model or sklearn model object
        """
        available_models = self.get_available_models()
        
        if isinstance(model_name, str):
            if model_name in available_models:
                self.model = available_models[model_name]
            else:
                raise ValueError(f"Model '{model_name}' not available. Choose from: {list(available_models.keys())}")
        else:
            # Assume it's a model object
            self.model = model_name
    
    def train_model(self, X_train=None, y_train=None, scale_features=False):
        """
        Train the selected model.
        
        Args:
            X_train: training features (uses self.X_train if None)
            y_train: training target (uses self.y_train if None)
            scale_features: bool, whether to scale features
        """
        if self.model is None:
            raise ValueError("No model selected. Use select_model() first.")
        
        if X_train is None or y_train is None:
            if self.X_train is None or self.y_train is None:
                raise ValueError("No training data available. Use split_data() first.")
            X_train, y_train = self.X_train, self.y_train
        
        # Feature scaling if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            if self.X_test is not None:
                self.X_test = self.scaler.transform(self.X_test)
        
        # Train the model
        try:
            self.model.fit(X_train, y_train)
            print(f"Model trained successfully: {type(self.model).__name__}")
        except Exception as e:
            raise RuntimeError(f"Error training model: {e}")
    
    def evaluate_model(self, X_test=None, y_test=None, detailed=True):
        """
        Evaluate the trained model.
        
        Args:
            X_test: test features (uses self.X_test if None)
            y_test: test target (uses self.y_test if None)
            detailed: bool, whether to return detailed metrics
        
        Returns:
            dict: evaluation metrics
        """
        if self.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        if X_test is None or y_test is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError("No test data available. Use split_data() first.")
            X_test, y_test = self.X_test, self.y_test
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test)
        
        if self.problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
            }
            
            if detailed:
                metrics.update({
                    'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
                    'classification_report': classification_report(y_test, predictions, output_dict=True)
                })
                
                # ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        if hasattr(self.model, 'predict_proba'):
                            y_proba = self.model.predict_proba(X_test)[:, 1]
                            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                        elif hasattr(self.model, 'decision_function'):
                            y_scores = self.model.decision_function(X_test)
                            metrics['roc_auc'] = roc_auc_score(y_test, y_scores)
                    except:
                        pass
        
        else:  # regression
            metrics = {
                'r2_score': r2_score(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions))
            }
            
            if detailed:
                # Calculate additional regression metrics
                ss_res = np.sum((y_test - predictions) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                adj_r2 = 1 - (ss_res / ss_tot) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
                metrics['adjusted_r2'] = adj_r2
                
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                metrics['mape'] = mape if not np.isinf(mape) else None
        
        return metrics
    
    def cross_validate(self, cv=5, scoring=None):
        """
        Perform cross-validation on the model.
        
        Args:
            cv: int, number of cross-validation folds
            scoring: string, scoring metric
        
        Returns:
            dict: cross-validation results
        """
        if self.model is None:
            raise ValueError("No model selected. Use select_model() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Use split_data() first.")
        
        X = self.X_train
        y = self.y_train
        
        # Apply scaling if needed
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)