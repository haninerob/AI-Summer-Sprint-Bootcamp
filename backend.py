from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
try:
    model = joblib.load("best_rf_model.pkl")
    print(f"✅ Model loaded successfully from best_rf_model.pkl")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

def validate_file_type(filepath):
    if not str(filepath).endswith(".csv"):
        raise ValueError("Invalid file type. Please upload a CSV file.")

def load_csv_from_string(csv_string):
    try:
        df = pd.read_csv(io.StringIO(csv_string))
        df.columns = df.columns.str.strip()  # normalize columns
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")

def validate_required_columns(df, required_columns, default_means):
    missing_cols = [col for col in required_columns if col not in df.columns]

    if len(missing_cols) >= 3:
        raise ValueError(f"Too many required columns are missing: {missing_cols}")

    if 0 < len(missing_cols) <= 2:
        print(f"Missing columns detected: {missing_cols}")
        print("Filling missing columns with default mean values…")
        for col in missing_cols:
            df[col] = default_means.get(col, 0.0)  # fallback to 0.0 if no mean specified
    return df

def validate_numeric_columns(df, numeric_columns):
    for col in numeric_columns:
        if col in df.columns:  # Only check if column exists
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    raise ValueError(f"Column '{col}' must be numeric.")

def validate_missing_values(df, default_means):
    # Check per row
    rows_with_many_missing = df.isnull().sum(axis=1) > 2
    if rows_with_many_missing.any():
        raise ValueError("One or more rows have more than 2 missing values. File rejected.")

    # Fill per column
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"Columns with missing values: {missing_cols}")
        print("Filling missing values with default means.")
        for col in missing_cols:
            default = default_means.get(col)
            if default is not None:
                df[col] = df[col].fillna(default)
            else:
                raise ValueError(f"No default mean provided for column: {col}")

    return df

def validate_row_count(df):
    if len(df) == 0:
        raise ValueError("CSV file contains no rows.")

def remove_duplicates(df):
    return df.drop_duplicates()

def validate_input_data(csv_string):
    """
    Validates user-provided CSV data step by step.
    Returns cleaned and ready DataFrame if valid.
    """
    # Define schema of required columns
    required_columns = [
        'Minimum Orbit Intersection',
        'Absolute Magnitude',
        'Avg_Diameter_KM',
        'Perihelion Distance',
        'Orbit Uncertainity',
        'Inclination'
    ]
    numeric_columns = required_columns

    default_means = {
        'Minimum Orbit Intersection': 0.4768977851980138,
        'Absolute Magnitude': 0.1591059336579516,
        'Avg_Diameter_KM': 0.13758845293463146,
        'Perihelion Distance': 0.0479985661169944,
        'Orbit Uncertainity': 0.04306290759519788,
        'Inclination': 0.04022213144882968
    }

    # Load CSV
    df = load_csv_from_string(csv_string)

    # Run validations
    df = validate_required_columns(df, required_columns, default_means)
    validate_numeric_columns(df, numeric_columns)
    df = validate_missing_values(df, default_means)
    validate_row_count(df)

    # Clean up
    df = remove_duplicates(df)

    final_columns = [col for col in required_columns if col in df.columns]
    df_ready = df[final_columns]

    print("Input data validated successfully.")
    return df_ready

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get CSV data from request
        if request.is_json:
            # Handle JSON data (backward compatibility)
            data = request.json.get('processedData', [])
            df = pd.DataFrame(data)
        else:
            # Handle CSV file upload
            if 'file' in request.files:
                file = request.files['file']
                csv_string = file.read().decode('utf-8')
            else:
                csv_string = request.data.decode('utf-8')
            
            # Validate and process the CSV data
            df = validate_input_data(csv_string)
        
        # Define required features for the model
        required_features = [
            "Minimum Orbit Intersection",
            "Absolute Magnitude",
            "Avg_Diameter_KM",
            "Perihelion Distance",
            "Orbit Uncertainity",
            "Inclination"
        ]
        
        # Ensure all required features are present
        for feature in required_features:
            if feature not in df.columns:
                raise ValueError(f"Required feature '{feature}' is missing after validation")
        
        # Make prediction
        predictions = model.predict(df[required_features]).tolist()
        
        # Return predictions and the cleaned data
        return jsonify({
            'predictions': predictions,
            'data': df[required_features].to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)