from flask import Flask, request, render_template, jsonify
from pickle import load
import numpy as np
import africastalking
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the model and scaler
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))

# Africa's Talking API credentials from .env
username = os.getenv('AT_USER_NAME')
api_key = os.getenv('AT_API_KEY')

# Initialize Africa's Talking
africastalking.initialize(username, api_key)
sms = africastalking.SMS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wine_model_prediction', methods=['POST'])
def wine_model_prediction():
    try:
        # Retrieve form data
        new_wine = np.array([[
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])
        ]])

        # Scale and predict
        new_wine_scaled = scaler.transform(new_wine)
        prediction = model.predict(new_wine_scaled)

        # Prepare result message
        result = "Good Wine" if prediction[0] == 1 else "Bad Wine"
        phone_number = request.form['phone_number']
        message = f"Wine Quality Prediction: {result}"

        # Send SMS
        response = sms.send(message, [phone_number])
        
        # Display result on the dashboard and confirm SMS sent
        return render_template('index.html', prediction=result, sms_response="SMS sent successfully!" if response else "Failed to send SMS.")
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
