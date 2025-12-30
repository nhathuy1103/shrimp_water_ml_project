import os
import sys
import pickle

import joblib  
import numpy as np
from datetime import date
from lunardate import LunarDate


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Lưu object (preprocessor, model, v.v.) vào file.
    Ưu tiên joblib (ổn định hơn pickle cho sklearn).
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # joblib dùng tốt cho sklearn pipelines/models
        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load object từ file.
    Thử joblib trước, nếu fail thì fallback pickle (để cứu các file cũ).
    """
    try:
        # 1) thử joblib
        return joblib.load(file_path)

    except Exception:
        # 2) fallback pickle (file cũ)
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Dùng cho BÀI TOÁN PHÂN LOẠI (classification).
    Trả về report dict.
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
            f1 = f1_score(y_test, y_test_pred, average="weighted")

            report[model_name] = {
                "accuracy": acc,
                "f1_score": f1,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    


# src/utils.py

from datetime import date
from lunardate import LunarDate

def get_notice():
    today = date.today()
    lunar = LunarDate.fromSolarDate(today.year, today.month, today.day)
    d = lunar.day

    if d in (4, 5, 6):
        return (
            f"🦐 Hôm nay ngày {d} âm lịch, bà con còn xổ vuông không ạ? "
            "Nếu nhà mình đã đóng cống rồi thì mình nhớ kiểm tra kỹ lưỡng nước trước khi giữ lại trong vuông nghen.",
            "warn"
        )

    if d in (18, 19, 20):
        return (
            f"🦐 Hôm nay ngày {d} âm lịch, nước lớn đó ạ. "
            "Bà con nhớ coi lại nước ngoài sông kỹ rồi hẵng bơm vô vuông nghen.",
            "warn"
        )

    return None, None
