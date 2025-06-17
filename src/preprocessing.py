import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

def load_data(file, na_values=None, keep_default_na=True, encoding='utf-8'):
    """
    Load a CSV or Excel file from a file path (string) or file-like object.
    
    Args:
        file: File path (string) or file-like object
        na_values: Additional values to recognize as NA/NaN
        keep_default_na: Whether to include default NA values
        encoding: Text encoding for CSV files
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    import os
    
    if hasattr(file, "name"):  # file-like object
        ext = os.path.splitext(file.name)[-1].lower()
    elif isinstance(file, str):  # string path
        ext = os.path.splitext(file)[-1].lower()
    else:
        raise TypeError("File must be a string path or file-like object with a 'name' attribute.")

    # Enhanced default NA values
    if na_values is None:
        na_values = [
            '', ' ', 'NA', 'N/A', 'na', 'n/a', 'null', 'NULL', 'None', 'NONE', '?', 
            '#N/A', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#NULL!',
            'nan', 'NaN', 'NAN', '-', '--', '---', 'missing', 'MISSING',
            'undefined', 'UNDEFINED', 'empty', 'EMPTY'
        ]

    try:
        if ext == '.csv':
            return pd.read_csv(file, na_values=na_values, keep_default_na=keep_default_na, encoding=encoding)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file, na_values=na_values, keep_default_na=keep_default_na)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    except UnicodeDecodeError:
        if ext == '.csv':
            try:
                return pd.read_csv(file, na_values=na_values, keep_default_na=keep_default_na, encoding='latin-1')
            except:
                return pd.read_csv(file, na_values=na_values, keep_default_na=keep_default_na, encoding='cp1252')
        else:
            raise
    except Exception as e:
        raise IOError(f"Failed to load file: {e}")

def get_data_info(data):
    """
    Get comprehensive information about the DataFrame.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        dict: Information about the DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    info = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'null_counts': data.isnull().sum().to_dict(),
        'null_percentages': (data.isnull().sum() / len(data) * 100).to_dict(),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': data.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    return info

def show_head(data, n=5):
    """Return the first n rows of the DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    return data.head(n)

def show_tail(data, n=5):
    """Return the last n rows of the DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    return data.tail(n)

def get_dtypes(data):
    """Return data types of each column in the DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    return data.dtypes

def get_unique_values(data, column):
    """
    Return the unique values for a given column or list of columns.
    
    Args:
        data: pandas DataFrame
        column: string or list of column names
    
    Returns:
        numpy array or dict: Unique values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    if isinstance(column, (list, tuple)):
        missing = [col for col in column if col not in data.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return {col: data[col].unique() for col in column}
    else:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        return data[column].unique()

def get_null_counts(data):
    """Return count of null (NaN) values in each column."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    return data.isnull().sum()

def handle_nulls(data, method="drop", columns=None, value=None, strategy='mean'):
    """
    Handle null values in the DataFrame with enhanced methods.
    
    Args:
        data: pandas DataFrame
        method: str - 'drop', 'ffill', 'bfill', 'mean', 'median', 'mode', 'custom', 'knn', 'interpolate'
        columns: list of column names or None for all columns
        value: custom value for 'custom' method
        strategy: strategy for KNN imputer ('mean', 'median', 'most_frequent', 'constant')
    
    Returns:
        pandas DataFrame: DataFrame with null values handled
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    if columns is None:
        cols = data.columns
    else:
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, (list, tuple)):
            raise TypeError("Columns must be a string or list/tuple of column names.")
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        cols = columns

    df = data.copy()

    if method == "drop":
        return df.dropna(subset=cols)
    
    elif method == "ffill":
        df[cols] = df[cols].fillna(method='ffill')
        return df
    
    elif method == "bfill":
        df[cols] = df[cols].fillna(method='bfill')
        return df
    
    elif method == "interpolate":
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
        for col in numeric_cols:
            df[col] = df[col].interpolate()
        return df
    
    elif method == "mean":
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
            else:
                raise TypeError(f"Mean imputation not supported for non-numeric column: {col}")
        return df
    
    elif method == "median":
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
            else:
                raise TypeError(f"Median imputation not supported for non-numeric column: {col}")
        return df
    
    elif method == "mode":
        for col in cols:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                mode_val = mode_vals[0]
                df[col].fillna(mode_val, inplace=True)
        return df
    
    elif method == "custom":
        if value is None:
            raise ValueError("Custom value must be provided when method='custom'.")
        for col in cols:
            df[col].fillna(value, inplace=True)
        return df
    
    elif method == "knn":
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5, weights='uniform')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df
    
    else:
        raise ValueError("Invalid method. Choose from 'drop', 'ffill', 'bfill', 'mean', 'median', 'mode', 'custom', 'knn', 'interpolate'.")

def convert_dtype(data, column, new_type):
    """
    Convert the data type of a DataFrame column to a new type.
    
    Args:
        data: pandas DataFrame
        column: string - column name
        new_type: string or type - new data type
    
    Returns:
        pandas DataFrame: DataFrame with converted column
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    df = data.copy()
    
    try:
        if new_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == 'category':
            df[column] = df[column].astype('category')
        else:
            df[column] = df[column].astype(new_type)
    except Exception as e:
        raise ValueError(f"Cannot convert column '{column}' to type '{new_type}': {e}")
    
    return df

def apply_encoding(data, columns, method="onehot", drop_first=True, handle_unknown='ignore'):
    """
    Apply encoding to categorical column(s) with enhanced options.
    
    Args:
        data: pandas DataFrame
        columns: list of column names to encode
        method: str - 'onehot', 'label', 'target' (for target encoding)
        drop_first: bool - whether to drop first column in one-hot encoding
        handle_unknown: str - how to handle unknown categories in one-hot encoding
    
    Returns:
        pandas DataFrame: DataFrame with encoded columns
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, (list, tuple)):
        raise TypeError("Columns must be a string or list/tuple of column names.")
    
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    df = data.copy()

    if method == "onehot":
        return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)
    
    elif method == "label":
        for col in columns:
            # Handle missing values before encoding
            if df[col].isnull().any():
                df[col] = df[col].fillna('missing')
            
            encoder = LabelEncoder()
            try:
                df[col] = encoder.fit_transform(df[col].astype(str))
            except Exception as e:
                raise ValueError(f"Error encoding column '{col}' with LabelEncoder: {e}")
        return df
    
    else:
        raise ValueError("Invalid encoding method. Choose 'onehot' or 'label'.")

def scale_data(data, columns=None, method='standard', feature_range=(0, 1)):
    """
    Scale numeric features using various scaling methods.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to scale (None for all numeric columns)
        method: str - 'standard', 'minmax', 'robust'
        feature_range: tuple - range for MinMaxScaler
    
    Returns:
        pandas DataFrame: DataFrame with scaled columns
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    if not isinstance(columns, (list, tuple)):
        raise TypeError("Columns must be a string or list/tuple of column names.")
    
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Check if columns are numeric
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns cannot be scaled: {non_numeric}")
    
    try:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'.")
        
        df[columns] = scaler.fit_transform(df[columns])
        return df
    
    except Exception as e:
        raise ValueError(f"Error scaling data: {e}")

def remove_outliers(data, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from numeric columns.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to check for outliers (None for all numeric)
        method: str - 'iqr' or 'zscore'
        threshold: float - threshold for outlier detection
    
    Returns:
        pandas DataFrame: DataFrame with outliers removed
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    if method == 'iqr':
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[z_scores < threshold]
    
    else:
        raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")
    
    return df

def create_feature_summary(data):
    """
    Create a comprehensive summary of features in the dataset.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        pandas DataFrame: Summary of features
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    summary_data = []
    
    for col in data.columns:
        col_info = {
            'Column': col,
            'Data_Type': str(data[col].dtype),
            'Non_Null_Count': data[col].count(),
            'Null_Count': data[col].isnull().sum(),
            'Null_Percentage': round(data[col].isnull().sum() / len(data) * 100, 2),
            'Unique_Values': data[col].nunique(),
            'Memory_Usage': data[col].memory_usage(deep=True)
        }
        
        if pd.api.types.is_numeric_dtype(data[col]):
            col_info.update({
                'Mean': round(data[col].mean(), 2) if not data[col].isnull().all() else None,
                'Std': round(data[col].std(), 2) if not data[col].isnull().all() else None,
                'Min': data[col].min() if not data[col].isnull().all() else None,
                'Max': data[col].max() if not data[col].isnull().all() else None,
            })
        else:
            col_info.update({
                'Most_Frequent': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                'Mean': None,
                'Std': None,
                'Min': None,
                'Max': None
            })
        
        summary_data.append(col_info)
    
    return pd.DataFrame(summary_data)