import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainer:

    def __init__(self):
        self.models = {
            "task1": RandomForestClassifier(n_estimators=300),
            "task2": XGBRegressor(n_estimators=400, max_depth=5),
            "task3": RandomForestClassifier(n_estimators=200),
            "task4": GradientBoostingClassifier()
        }

    def train_all_tasks(self, data_dict):
        try:
            results = {}

            for task, (X_train, y_train, X_test, y_test) in data_dict.items():

                model = self.models[task]
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                if task == "task2":
                    score = r2_score(y_test, preds)
                else:
                    score = accuracy_score(y_test, preds)

                model_path = os.path.join("artifacts", f"model_{task}.pkl")
                save_object(model_path, model)

                results[task] = score
                logging.info(f"{task} score: {score}")

            return results

        except Exception as e:
            raise CustomException(e, sys)
