from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and the label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Render the form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        precipitation = float(request.form['precipitation'])
        max_temp = float(request.form['max_temp'])
        min_temp = float(request.form['min_temp'])
        wind_speed = float(request.form['wind_speed'])

        # Prepare the feature array
        features = np.array([precipitation, max_temp, min_temp, wind_speed]).reshape(1, -1)

        # Make the prediction using the model
        prediction = model.predict(features)[0]  # Predict returns a numpy array, we take the first value

        # Decode the prediction label back to the weather condition
        weather_prediction = label_encoder.inverse_transform([prediction])[0]

        # Return the decoded weather prediction as a response, render output.html
        return render_template('output.html', prediction=weather_prediction)

    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
