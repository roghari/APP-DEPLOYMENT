from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Load the dataset from CSV
file_path = "C:/Users/harih/OneDrive/Desktop/FAPP/city_hour.csv"
df = pd.read_csv(file_path)
df = df.dropna()

# Select only the relevant attributes and the target variable
selected_features = ['Pma', 'PM', 'Nitrogen_dioxide', 'CO', 'Sulfur_dioxide', 'Ozone']
target = 'target'
selected_data = df[selected_features + [target]]

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(selected_data[selected_features], selected_data[target])

# Save the trained model
joblib.dump(model, 'linear_regression_model.pkl')

# Load the trained model
loaded_model = joblib.load('linear_regression_model.pkl')

# Define input range values
input_ranges = {
    'Pma': (0, 250),
    'PM': (0, 425),
    'Nitrogen_dioxide': (0, 1251),
    'Ozone': (0, 451),
    'Sulfur_dioxide': (0, 605),
    'CO': (0, 30.5)
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        error_message = None

        for feature, (lower, upper) in input_ranges.items():
            while True:
                try:
                    value = float(request.form[feature])
                    if lower <= value <= upper:
                        input_data[feature] = value
                        break
                    else:
                        error_message = f"Error: {feature} must be in the range of {lower}-{upper}"
                        break
                except ValueError:
                    error_message = f"Error: {feature} must be a numeric value."
                    break

        if error_message:
            return render_template('index.html', error_message=error_message)

        # Make predictions using the loaded model
        input_values = [input_data[feature] for feature in selected_features]
        input_data_array = np.array([input_values])

        # Classify air quality based on the predicted value
        prediction = loaded_model.predict(input_data_array)[0]

        # Define air quality categories and corresponding thresholds
        thresholds = {
            'Good': 50,
            'Moderate': 100,
            'Unhealthy for Sensitive Groups': 150,
            'Unhealthy': 200,
            'Very Unhealthy': 300,
            'Hazardous': 500
        }

        # Classify air quality based on the predicted value
        air_quality_category = None
        for category, threshold in thresholds.items():
            if prediction <= threshold:
                air_quality_category = category
                break

        return render_template('result.html', prediction=prediction, air_quality_category=air_quality_category)

    except Exception as e:
        error_message = f"Error occurred: {e}"
        return render_template('index.html', error_message=error_message)

