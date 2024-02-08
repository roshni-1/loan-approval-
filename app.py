from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('loan_approval_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        credit_score = float(request.form['credit_score'])
        relationship_duration = float(request.form['relationship_duration'])
        repayment_tenure = float(request.form['repayment_tenure'])
        income = float(request.form['income'])
        investments = float(request.form['investments'])
        savings_account_balance = float(request.form['savings_account_balance'])
        outstanding_loan = float(request.form['outstanding_loan'])

        input_data = np.array([[credit_score, relationship_duration, repayment_tenure, income, investments,
                                savings_account_balance, outstanding_loan]])
        prediction = model.predict(input_data)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
