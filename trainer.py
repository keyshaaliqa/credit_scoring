import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from src.evaluator import Evaluator


class ModelTrainer:

    def __init__(self, preprocessor):

        self.preprocessor = preprocessor
        self.evaluator = Evaluator()

    def train_model(
        self,
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        pipeline = Pipeline([("preprocessor", self.preprocessor), ("model", model)])

        with mlflow.start_run(run_name=model_name):

            pipeline.fit(X_train, y_train)

            pred = pipeline.predict(X_test)

            result = self.evaluator.evaluate(y_test, pred)

            mlflow.log_param("model", model_name)

            for k, v in result.items():

                if k != "confusion_matrix":
                    mlflow.log_metric(k, v)

            disp = ConfusionMatrixDisplay(confusion_matrix=result["confusion_matrix"])

            disp.plot()

            plt.savefig("confusion_matrix.png")

            mlflow.log_artifact("confusion_matrix.png")

            try:
                mlflow.sklearn.log_model(
                    pipeline,
                    "model",
                    skops_trusted_types=[
                        "numpy.dtype",
                        "sklearn.compose._column_transformer._RemainderColsList",
                        "sklearn.pipeline.Pipeline",
                        "sklearn.compose._column_transformer.ColumnTransformer",
                        "sklearn.preprocessing._data.StandardScaler",
                        "sklearn.impute._base.SimpleImputer",
                        "sklearn.preprocessing._encoders.OneHotEncoder",
                        "sklearn.linear_model._logistic.LogisticRegression",
                        "sklearn.tree._classes.DecisionTreeClassifier",
                        "sklearn.ensemble._forest.RandomForestClassifier"
                    ]
                )
            except TypeError as exc:
                if "skops_trusted_types" in str(exc):
                    mlflow.sklearn.log_model(pipeline, "model")
                else:
                    raise

            return pipeline, result

    def train_logistic(self, X_train, X_test, y_train, y_test):

        return self.train_model(
            "LogisticRegression",
            LogisticRegression(max_iter=1000),
            X_train,
            X_test,
            y_train,
            y_test
        )

    def train_decision_tree(self, X_train, X_test, y_train, y_test):

        return self.train_model(
            "DecisionTree",
            DecisionTreeClassifier(random_state=42),
            X_train,
            X_test,
            y_train,
            y_test
        )

    def train_random_forest(self, X_train, X_test, y_train, y_test):

        return self.train_model(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )