import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer


class Preprocessor:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):

        self.df = pd.read_csv(self.filepath)
        return self.df

    def clean_data(self):

        drop_cols = [
            "Unnamed: 0",
            "ID",
            "Customer_ID",
            "Name",
            "SSN",
            "Credit_History_Age"
        ]

        self.df.drop(columns=drop_cols, inplace=True)

        # Mendefinisikan kolom numerik secara eksplisit untuk langkah pembersihan
        numeric_cols_for_cleaning = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        for col in numeric_cols_for_cleaning:
            # Konversi ke numerik, memaksa error (string non-numerik) menjadi NaN
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        return self.df

    def transform(self):

        numeric = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        categorical = [
            "Month",
            "Occupation",
            "Type_of_Loan",
            "Credit_Mix",
            "Payment_of_Min_Amount",
            "Payment_Behaviour"
        ]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical)
        ])

        return self.preprocessor

    def split_data(self):

        X = self.df.drop("Credit_Score", axis=1)
        y = self.df["Credit_Score"]

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )