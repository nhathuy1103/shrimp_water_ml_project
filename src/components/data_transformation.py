import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    @staticmethod
    def _add_domain_labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def _label_vibrio(row):
            vib = row.get("VIBRIO_TONG_SO_CFU_ML", np.nan)
            para = row.get("VIBRIO_PARAHAEMOLYTICUS_CFU_ML", 0.0)
            do = row.get("DO", np.nan)

            risk_score = 0

            try:
                vib_val = float(vib)
            except Exception:
                vib_val = np.nan

            if not np.isnan(vib_val):
                if vib_val >= 1e4:
                    risk_score += 2
                elif vib_val >= 1e3:
                    risk_score += 1

            try:
                para_val = float(para)
            except Exception:
                para_val = 0.0
            if para_val > 0:
                risk_score += 1

            try:
                do_val = float(do)
            except Exception:
                do_val = np.nan
            if not np.isnan(do_val) and do_val < 4:
                risk_score += 1

            if risk_score <= 1:
                return 0  # THAP
            elif risk_score == 2:
                return 1  # TRUNG_BINH
            else:
                return 2  # CAO

        df["LABEL_VIBRIO_RISK"] = df.apply(_label_vibrio, axis=1)

        def _label_tao(row):
            do_trong = row.get("DO_TRONG", np.nan)
            po43 = row.get("PO43", np.nan)
            no3 = row.get("NO3", np.nan)
            cod = row.get("COD", np.nan)

            def _to_float(v):
                try:
                    return float(v)
                except Exception:
                    return np.nan

            do_trong = _to_float(do_trong)
            po43 = _to_float(po43)
            no3 = _to_float(no3)
            cod = _to_float(cod)

            # Thiếu tảo: nước quá trong hoặc dinh dưỡng rất thấp
            if (not np.isnan(do_trong) and do_trong > 50) or (
                (np.isnan(po43) or po43 < 0.05)
                and (np.isnan(no3) or no3 < 0.1)
            ):
                return 0  # THIEU_TAO

            # Cân bằng
            if (
                not np.isnan(do_trong)
                and 25 <= do_trong <= 40
                and (np.isnan(po43) or 0.05 <= po43 <= 0.3)
                and (np.isnan(cod) or cod <= 30)
            ):
                return 1  # CAN_BANG

            # Còn lại: nguy cơ nở hoa tảo
            return 2  # NGUY_CO_NO_HOA

        df["LABEL_TAO_THUC_AN"] = df.apply(_label_tao, axis=1)

        # 3. Nhãn chất lượng nước tổng hợp
        def _label_water(row):
            def to_f(v):
                try:
                    return float(v)
                except Exception:
                    return np.nan

            nhiet_do = to_f(row.get("NHIET_DO"))
            ph = to_f(row.get("PH"))
            do = to_f(row.get("DO"))
            do_man = to_f(row.get("DO_MAN"))
            do_kiem = to_f(row.get("DO_KIEM"))
            no2 = to_f(row.get("NO2"))
            nh4 = to_f(row.get("NH4"))
            cod = to_f(row.get("COD"))

            score = 0

            if not np.isnan(nhiet_do) and not (27 <= nhiet_do <= 32):
                score += 1
            if not np.isnan(ph) and not (7.5 <= ph <= 8.5):
                score += 1
            if not np.isnan(do) and do < 4:
                score += 1
            if not np.isnan(do_man) and not (10 <= do_man <= 30):
                score += 1
            if not np.isnan(do_kiem) and not (80 <= do_kiem <= 180):
                score += 1
            if not np.isnan(no2) and no2 > 0.3:
                score += 1
            if not np.isnan(nh4) and nh4 > 0.5:
                score += 1
            if not np.isnan(cod) and cod > 30:
                score += 1

            if score == 0:
                return 0  # TOT
            elif score <= 2:
                return 1  # CAN_THEO_DOI
            else:
                return 2  # NGUY_CO

        df["LABEL_WATER_QUALITY"] = df.apply(_label_water, axis=1)

        # TARGET cho model đầu tiên: rủi ro Vibrio
        df["TARGET"] = df["LABEL_VIBRIO_RISK"]

        return df


    def get_data_transformer_object(self):
        try:
            logging.info("Tạo đối tượng tiền xử lý dữ liệu (preprocessor) cho bộ dữ liệu tôm.")

            numeric_features = [
                "NHIET_DO",
                "PH",
                "DO",
                "DO_MAN",
                "DO_TRONG",
                "DO_KIEM",
                "NO2",
                "NO3",
                "NH4",
                "PO43",
                "COD",
                "VIBRIO_TONG_SO_CFU_ML",
                "VIBRIO_PARAHAEMOLYTICUS_CFU_ML",
                "VIBRIO_LOG",
                "NAM",
                "THANG",
                "NGAY",
            ]

            categorical_features = ["DIEM_QUAN_TRAC", "XA", "HUYEN"]

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    (
                        "scaler",
                        StandardScaler(with_mean=False),
                    ),
                ]
            )

            logging.info(f"Cột numeric: {numeric_features}")
            logging.info(f"Cột categorical: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numeric_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Đọc dữ liệu train và test cho bước Data Transformation.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Gắn nhãn domain (Vibrio risk, tảo, chất lượng nước).")
            train_df = self._add_domain_labels(train_df)
            test_df = self._add_domain_labels(test_df)

            target_column_name = "TARGET"

            # Các cột không nên đưa vào X (label & target)
            label_columns = [
                "LABEL_VIBRIO_RISK",
                "LABEL_TAO_THUC_AN",
                "LABEL_WATER_QUALITY",
                "TARGET",
            ]

            # Lấy X, y
            logging.info("Tách X, y cho train/test.")
            X_train_df = train_df.drop(columns=label_columns, axis=1)
            y_train = train_df[target_column_name]

            X_test_df = test_df.drop(columns=label_columns, axis=1)
            y_test = test_df[target_column_name]

            logging.info("Khởi tạo preprocessor.")
            preprocessor = self.get_data_transformer_object()

            logging.info("Fit & transform trên tập train, transform trên tập test.")
            X_train_arr = preprocessor.fit_transform(X_train_df)
            X_test_arr = preprocessor.transform(X_test_df)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info(
                f"Lưu preprocessor vào: {self.data_transformation_config.preprocessor_obj_file_path}"
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            logging.info("Hoàn thành bước Data Transformation.")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
