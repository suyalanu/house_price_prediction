from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('flask/models/training.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input data
    bedroom = int(request.form['bedroom'])
    bathroom = int(request.form['bathroom'])
    parking = int(request.form['parking'])
    area = float(request.form['area'])
    road_width = float(request.form['road_width'])
    build_area = float(request.form['build_area'])
    
    # Create features array
    features = np.array([[bedroom, bathroom, parking, area, road_width, build_area]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Pass the prediction as a number to be formatted in the template
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
