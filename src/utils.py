import os
import sys
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Lưu object (preprocessor, model, v.v.) vào file pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Dùng cho BÀI TOÁN PHÂN LOẠI (classification).
    Trả về:
    {
        "RandomForest": accuracy,
        "XGBoost": accuracy,
        ...
    }
    """

    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            para = param[model_name]

            gs = GridSearchCV(
                model,
                para,
                cv=3,
                scoring="accuracy",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_test_pred)
            f1  = f1_score(y_test, y_test_pred, average="weighted")

            report[model_name] = {
                "accuracy": acc,
                "f1_score": f1,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load object từ file pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
