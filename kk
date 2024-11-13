from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and the label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.form
        features = [float(data['precipitation']), float(data['max_temp']),
                    float(data['min_temp']), float(data['wind_speed'])]

        # Ensure the features are in the correct format (numpy array)
        features = np.array(features).reshape(1, -1)

        # Make the prediction using the model
        prediction = model.predict(features)[0]

        # Decode the prediction label back to the weather condition
        weather_prediction = label_encoder.inverse_transform([prediction])[0]

        # Render the output.html template and pass the decoded prediction
        return render_template('output.html', prediction=weather_prediction)

    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
