from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained machine learning modelfrom flask import Flask, render_template, request, jsonify
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
        # Extract input values from the JSON data
        data = request.json
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
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

model = joblib.load('loan_approval_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the request
        credit_score = float(request.form['credit_score'])
        relationship_duration = float(request.form['relationship_duration'])
        repayment_tenure = float(request.form['repayment_tenure'])
        income = float(request.form['income'])
        investments = float(request.form['investments'])
        savings_account_balance = float(request.form['savings_account_balance'])
        outstanding_loan = float(request.form['outstanding_loan'])

        # Make a prediction using the loaded model
        input_data = np.array([[credit_score, relationship_duration, repayment_tenure, income, investments,
                                savings_account_balance, outstanding_loan]])
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Return an error message as JSON
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
