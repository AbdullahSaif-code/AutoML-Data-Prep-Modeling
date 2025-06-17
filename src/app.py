import streamlit as st
import pandas as pd
import preprocessing
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Step-by-Step Data Preprocessing & Modeling", layout="wide")
st.title("Step-by-Step Data Preprocessing & Modeling")

st.markdown("#### 1. Upload your data file")
uploaded_file = st.file_uploader(
    "Upload your data file (CSV, XLSX, XLS)", 
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (.xlsx, .xls)"
)

if st.button("Try Example Data"):
    st.session_state['data'] = pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": ["x", "y", "z", None],
        "C": [10, None, 30, 40]
    })
    st.success("Loaded example data!")

if uploaded_file or 'data' in st.session_state:
    if uploaded_file and 'data' not in st.session_state:
        df = preprocessing.load_data(uploaded_file)
        null_percent = df.isnull().mean()
        dropped_cols = null_percent[null_percent > 0.5].index.tolist()
        if dropped_cols:
            df = df.drop(columns=dropped_cols)
            st.warning(f"Columns dropped due to >50% null values: {', '.join(dropped_cols)}")
        st.session_state['data'] = df

    data = st.session_state['data']

    st.markdown("---")
    st.markdown("#### 2. Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.selectbox("Rows to show", [5, 10, 15, 20], index=0)
    with col2:
        st.write(f"Total Rows: {data.shape[0]}")

    st.write("**Head of the data:**")
    st.write(data.head(num_rows))
    st.write("**Tail of the data:**")
    st.write(data.tail(num_rows))

    st.markdown("---")
    st.markdown("#### 3. Column Data Types and Unique Values")
    table_data = []
    for col in data.columns:
        unique_vals = data[col].unique()
        unique_display = ', '.join(map(str, unique_vals[:10]))
        if len(unique_vals) > 10:
            unique_display += ' ...'
        table_data.append({
            "Column": col,
            "Data Type": str(data[col].dtype),
            "Unique Values (first 10)": unique_display
        })
    st.table(pd.DataFrame(table_data))

    st.markdown("---")
    st.markdown("#### 4. Handle Null Values")
    st.info("Standard, error-based, and unstandardized nulls (e.g. NA, #N/A, missing, error, etc.) are automatically treated as null values.")
    st.write("**Null Value Summary:**")
    st.write(data.isnull().sum())

    null_strategy = st.radio(
        "Select how to handle null values:",
        ("drop", "ffill", "mean", "median", "mode"),
        help="Choose a strategy to handle missing values."
    )
    columns = st.multiselect(
        "Select columns to apply (leave empty for all):",
        data.columns.tolist()
    )
    columns = columns if columns else None

    custom_na = st.text_input(
        "Custom null/error values (comma separated, e.g. 'missing,error,not available')",
        value=""
    )
    custom_na_list = [v.strip() for v in custom_na.split(",") if v.strip()]

    if st.button("Apply Null Handling"):
        processed_data = preprocessing.handle_null_values(
            st.session_state['data'].copy(),
            strategy=null_strategy,
            columns=columns
        )
        st.session_state['data'] = processed_data
        st.success("Null handling applied!")
        st.write(processed_data.head())
        data = processed_data

    st.markdown("---")
    st.markdown("#### 5. Change Data Types")
    col_types = {}
    for col in st.session_state['data'].columns:
        dtype = st.selectbox(f"Select type for {col}", ["No Change", "int", "float", "str", "category"], key=col)
        if dtype != "No Change":
            col_types[col] = dtype
    if st.button("Apply Data Type Changes"):
        processed_data = preprocessing.change_data_types(
            st.session_state['data'].copy(),
            col_types
        )
        st.session_state['data'] = processed_data
        st.success("Data types changed!")
        st.write(processed_data.head())
        data = processed_data

    st.markdown("---")
    st.markdown("#### 6. Apply Encoding")
    cat_cols = [col for col in st.session_state['data'].columns if st.session_state['data'][col].dtype == 'object']
    encode_cols = st.multiselect("Select columns to encode", cat_cols)
    if st.button("Apply Encoding"):
        processed_data = preprocessing.apply_encoding(
            st.session_state['data'].copy(),
            encode_cols
        )
        st.session_state['data'] = processed_data
        st.success("Encoding applied!")
        st.write(processed_data.head())
        data = processed_data

    st.markdown("---")
    st.markdown("#### 7. Model Training & Export")
    data = st.session_state.get('data', data)
    if data is not None:
        st.write("Select target column for prediction:")
        target_column = st.selectbox("Target Column", data.columns)
        feature_columns = st.multiselect(
            "Select feature columns (leave empty for all except target):",
            [col for col in data.columns if col != target_column]
        )
        if not feature_columns:
            feature_columns = [col for col in data.columns if col != target_column]

        problem_type = st.radio("Problem Type", ["classification", "regression"])
        if problem_type == "classification":
            model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"], help="Choose a classification model.")
        else:
            model_name = st.selectbox("Select Model", ["Random Forest", "Linear Regression", "SVR"], help="Choose a regression model.")

        if st.button("Train Model"):
            if not np.all([np.issubdtype(data[feature].dtype, np.number) for feature in feature_columns]):
                st.error("All selected feature columns must be numeric. Please encode categorical columns.")
            elif not np.issubdtype(data[target_column].dtype, np.number):
                st.error("Target column must be numeric for regression. Please encode or select a numeric column.")
            else:
                from model_training import ModelTrainer
                trainer = ModelTrainer(data, target_column, problem_type)
                trainer.select_model(model_name)
                X_train, X_test, y_train, y_test = trainer.split_data(feature_columns)
                trainer.train_model(X_train, y_train)
                score = trainer.evaluate_model(X_test, y_test)

                predictions = trainer.model.predict(X_test)

                if problem_type == "regression":
                    mse = mean_squared_error(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    st.info(f"R² Score: {score:.4f}")
                    st.info(f"Mean Squared Error (MSE): {mse:.4f}")
                    st.info(f"Mean Absolute Error (MAE): {mae:.4f}")
                else:
                    st.info(f"Accuracy: {score:.4f}")

                results_df = pd.DataFrame({
                    "Actual": y_test,
                    "Predicted": predictions
                })
                results_df["Original Row"] = y_test.index
                results_df = results_df.reset_index(drop=True)
                results_df = results_df[["Original Row", "Actual", "Predicted"]]

                st.subheader("Test Set Results")
                st.write(results_df.head(20))

                if problem_type == "regression":
                    st.line_chart(results_df)

                st.session_state['trained_model'] = trainer.model

    if 'trained_model' in st.session_state:
        file_format = st.selectbox("Export format", ["joblib", "pickle"])
        model_filename = st.text_input("Model file name", value="my_model")
        if st.button("Export Model"):
            from export import export_model, export_summary
            export_model(st.session_state['trained_model'], model_filename, file_format)
            st.success(export_summary(model_filename, file_format))
            
            if file_format == "joblib":
                file_path = f"{model_filename}.joblib"
            else:
                file_path = f"{model_filename}.pkl"
            
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download Model File",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="application/octet-stream"
                )

    st.markdown("---")
    st.markdown("### Summary")
    st.write(f"Final dataset shape: {st.session_state['data'].shape}")
    st.write("Columns:", list(st.session_state['data'].columns))
    csv = st.session_state['data'].to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed Data", csv, "processed_data.csv", "text/csv")
