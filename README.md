# README for Stermlit Flask App

## Overview
The Stermlit Flask App is a web application designed to facilitate data preprocessing, model training, and model exportation using a user-friendly interface. This application leverages the Flask framework to provide a seamless experience for users looking to perform machine learning tasks.

## Project Structure
```
stermlit-flask-app
├── src
│   ├── app.py               # Main entry point of the Flask application
│   ├── preprocing.py        # Functions for data preprocessing
│   ├── model-traning.py     # Model training functionalities
│   ├── export.py            # Export trained models
│   ├── templates
│   │   └── index.html       # HTML template for the web UI
│   └── static
│       └── style.css        # CSS styles for the web application
├── requirements.txt         # List of dependencies
└── README.md                # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd stermlit-flask-app
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
1. Run the Flask application:
   ```
   python src/app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

## Features
- **Data Preprocessing**: Load datasets, view data summaries, handle null values, and apply encoding.
- **Model Training**: Select features and target variables, choose problem types, and evaluate model performance.
- **Model Export**: Export trained models in various formats as per user requirements.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.