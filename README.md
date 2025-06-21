# README for Streamlit AutoML Data Prep & Modeling App

## Overview
This Streamlit application provides an interactive interface for step-by-step data preprocessing, model training, and model export. It is designed for users who want to quickly prepare data and build machine learning models without writing code.

## Project Structure
```
AutoML-Data-Prep-Modeling
├── src
│   ├── app.py               # Main Streamlit app
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── model_training.py    # Model training functionalities
│   ├── export.py            # Export trained models
│   ├── config.toml          # Configuration file
│   └── __pycache__/         # Python cache files
├── requirements.txt         # List of dependencies
└── README.md                # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/AbdullahSaif-code/AutoML-Data-Prep-Modeling.git
   cd AutoML-Data-Prep-Modeling
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. The app will open in your default web browser, or you can navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## Features
- **Data Preprocessing**: Upload datasets, view data summaries, handle null values, change data types, and apply encoding.
- **Model Training**: Select features and target variables, choose problem types (classification/regression), train models, and view evaluation metrics.
- **Model Export**: Export trained models in `joblib` or `pickle` format and download processed data.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.