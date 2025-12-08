import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        self.numeric_features = [
            "NHIET_DO", "PH", "DO",
            "DO_MAN", "DO_TRONG", "DO_KIEM",
            "NO2", "NO3", "NH4", "PO43",
            "COD",
            "NAM", "THANG", "NGAY"
        ]

        self.categorical_features = ["DIEM_QUAN_TRAC", "XA", "HUYEN"]


    @staticmethod
    def label_vibrio_risk(v):
        if v < 4:
            return 0
        elif v < 6:
            return 1
        else:
            return 2

    def add_labels(self, df: pd.DataFrame):

        df["LABEL_VIBRIO_RISK"] = df["VIBRIO_LOG"].apply(self.label_vibrio_risk)

        df["LABEL_MOI_TRUONG_TOM"] = (
            (df["NHIET_DO"].between(26, 30)) &
            (df["PH"].between(7.5, 8.5)) &
            (df["DO"] >= 4) &
            (df["DO_MAN"].between(5, 30)) &
            (df["NO2"] <= 0.25) &
            (df["NH4"] <= 0.3)
        ).astype(int)

        df["LABEL_TAO_THUC_AN"] = (
            (df["DO_TRONG"].between(25, 40)) &
            (df["PO43"].between(0.03, 0.5)) &
            (df["NO3"].between(0.05, 1.5))
        ).astype(int)

        return df


    def get_preprocessor(self):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("scaler", StandardScaler(with_mean=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.numeric_features),
            ("cat", cat_pipeline, self.categorical_features)
        ])

        return preprocessor


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self.add_labels(train_df)
            test_df = self.add_labels(test_df)

            train_df = train_df.sort_values(["NAM", "THANG", "NGAY"])
            test_df = test_df.sort_values(["NAM", "THANG", "NGAY"])

            X_train = train_df[self.numeric_features + self.categorical_features]
            X_test = test_df[self.numeric_features + self.categorical_features]

            preprocessing_obj = self.get_preprocessor()
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)

            data_dict = {}

            data_dict["task1"] = (
                X_train_arr,
                train_df["LABEL_VIBRIO_RISK"].values,
                X_test_arr,
                test_df["LABEL_VIBRIO_RISK"].values,
            )

            data_dict["task2"] = (
                X_train_arr,
                train_df["VIBRIO_LOG"].values,
                X_test_arr,
                test_df["VIBRIO_LOG"].values,
            )

            data_dict["task3"] = (
                X_train_arr,
                train_df["LABEL_MOI_TRUONG_TOM"].values,
                X_test_arr,
                test_df["LABEL_MOI_TRUONG_TOM"].values,
            )

            data_dict["task4"] = (
                X_train_arr,
                train_df["LABEL_TAO_THUC_AN"].values,
                X_test_arr,
                test_df["LABEL_TAO_THUC_AN"].values,
            )

            return data_dict

        except Exception as e:
            raise CustomException(e, sys)
