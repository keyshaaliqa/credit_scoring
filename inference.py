import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

sample = pd.read_csv("data_B.csv")

sample = sample.iloc[:5]

prediction = model.predict(sample)

print(prediction)