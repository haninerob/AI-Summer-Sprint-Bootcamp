from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json['processedData']
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Define required features for the model
        required_features = [
            "Minimum Orbit Intersection",
            "Absolute Magnitude",
            "Avg_Diameter_KM",
            "Perihelion Distance",
            "Orbit Uncertainity",
            "Inclination"
        ]
        
        # Make prediction
        predictions = model.predict(df[required_features]).tolist()
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)