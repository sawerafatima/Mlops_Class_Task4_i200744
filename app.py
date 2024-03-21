from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('svm_model.pkl')

# Define API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.json
    
    # Convert input data to numpy array
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Map class index to class label
    class_label = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_class = class_label[prediction[0]]
    
    # Return prediction as JSON response
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True)
