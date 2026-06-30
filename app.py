from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# Load preprocessing
preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {

        "Age": float(request.form["Age"]),
        "Annual_Income": float(request.form["Annual_Income"]),
        "Monthly_Inhand_Salary": float(request.form["Monthly_Inhand_Salary"]),
        "Num_Bank_Accounts": int(request.form["Num_Bank_Accounts"]),
        "Num_Credit_Card": int(request.form["Num_Credit_Card"]),
        "Interest_Rate": float(request.form["Interest_Rate"]),
        "Num_of_Loan": int(request.form["Num_of_Loan"]),
        "Delay_from_due_date": int(request.form["Delay_from_due_date"]),
        "Num_of_Delayed_Payment": int(request.form["Num_of_Delayed_Payment"]),
        "Changed_Credit_Limit": float(request.form["Changed_Credit_Limit"]),
        "Num_Credit_Inquiries": int(request.form["Num_Credit_Inquiries"]),
        "Outstanding_Debt": float(request.form["Outstanding_Debt"]),
        "Credit_Utilization_Ratio": float(request.form["Credit_Utilization_Ratio"]),
        "Total_EMI_per_month": float(request.form["Total_EMI_per_month"]),
        "Amount_invested_monthly": float(request.form["Amount_invested_monthly"]),
        "Monthly_Balance": float(request.form["Monthly_Balance"]),

        "Month": request.form["Month"],
        "Occupation": request.form["Occupation"],
        "Type_of_Loan": request.form["Type_of_Loan"],
        "Credit_Mix": request.form["Credit_Mix"],
        "Payment_of_Min_Amount": request.form["Payment_of_Min_Amount"],
        "Payment_Behaviour": request.form["Payment_Behaviour"]
    }

    input_data = pd.DataFrame([data])

    input_data = preprocessor.transform(input_data)

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)