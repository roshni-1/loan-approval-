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
        # Extract input values from the JSON request
        data = request.get_json()

        # Validate input fields
        required_fields = ['credit_score', 'relationship_duration', 'repayment_tenure', 'income',
                            'investments', 'savings_account_balance', 'outstanding_loan']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Convert input values to float
        credit_score = float(data['credit_score'])
        relationship_duration = float(data['relationship_duration'])
        repayment_tenure = float(data['repayment_tenure'])
        income = float(data['income'])
        investments = float(data['investments'])
        savings_account_balance = float(data['savings_account_balance'])
        outstanding_loan = float(data['outstanding_loan'])

        # Make a prediction using the loaded model
        input_data = np.array([[credit_score, relationship_duration, repayment_tenure, income, investments,
                                savings_account_balance, outstanding_loan]])
        prediction = int(model.predict(input_data)[0])

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
