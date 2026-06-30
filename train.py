import joblib
import pandas as pd
import os

from src.preprocessing import Preprocessor
from src.trainer import ModelTrainer

pre = Preprocessor("data_B.csv")

pre.load_data()
pre.clean_data()

processor = pre.transform()

X_train, X_test, y_train, y_test = pre.split_data()

trainer = ModelTrainer(processor)

models = {}

models["Logistic"] = trainer.train_logistic(
    X_train,
    X_test,
    y_train,
    y_test
)

models["DecisionTree"] = trainer.train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test
)

models["RandomForest"] = trainer.train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)

best_name = None
best_score = 0

for name, (_, metric) in models.items():

    if metric["f1"] > best_score:

        best_score = metric["f1"]
        best_name = name

# FIX: Pastikan direktori 'models' ada sebelum menyimpan file
os.makedirs("models", exist_ok=True)

joblib.dump(
    models[best_name][0],
    "models/best_model.pkl"
)

print("Best Model :", best_name)
print("Best F1 :", best_score)