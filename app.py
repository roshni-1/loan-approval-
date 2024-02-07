from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('loan_approval_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the request
        data = request.get_json()

        # Validate if all required fields are present
        if all(key in data for key in ['credit_score', 'relationship_duration', 'repayment_tenure',
                                       'income', 'investments', 'savings_account_balance', 'outstanding_loan']):
            # Make a prediction using the loaded model
            input_data = np.array([[float(data['credit_score']), float(data['relationship_duration']),
                                    float(data['repayment_tenure']), float(data['income']),
                                    float(data['investments']), float(data['savings_account_balance']),
                                    float(data['outstanding_loan'])]])

            prediction = model.predict(input_data)

            # Return the prediction as JSON
            return jsonify({'prediction': int(prediction[0])})
        else:
            # If any required field is missing, return an error
            return jsonify({'error': 'Missing required fields'}), 400

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
