import os
import sys
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import accuracy_score, r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainer:

    def __init__(self):


        # Task 1: Phân loại nguy cơ Vibrio
        gb_task1 = GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=100,
            subsample=0.9,
            random_state=42,
        )

        # Task 2: Hồi quy VIBRIO_LOG
        gb_task2 = GradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=1,
            n_estimators=300,
            subsample=0.9,
            random_state=42,
        )

        # Task 3: Đánh giá chất lượng môi trường ao nuôi
        gb_task3 = GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            n_estimators=100,
            subsample=0.9,
            random_state=42,
        )

        # Task 4: Đánh giá điều kiện phát triển của tảo
        gb_task4 = GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            n_estimators=80,
            subsample=0.6,
            random_state=42,
        )

        self.models = {
            "task1": gb_task1,
            "task2": gb_task2,
            "task3": gb_task3,
            "task4": gb_task4,
        }

    def train_all_tasks(self, data_dict):
        try:
            results = {}

            for task, (X_train, y_train, X_test, y_test) in data_dict.items():
                logging.info(f"Training model for {task}...")

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
                logging.info(f"{task} model saved to: {model_path}")

            return results

        except Exception as e:
            raise CustomException(e, sys)
