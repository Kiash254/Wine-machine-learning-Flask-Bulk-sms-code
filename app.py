from flask import Flask, request, render_template, jsonify
from pickle import load
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wine_model_prediction', methods=['POST'])
def wine_model_prediction():
    # Retrieve data from the form
    try:
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

        # Return result
        result = "Good Wine" if prediction[0] == 1 else "Bad Wine"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
