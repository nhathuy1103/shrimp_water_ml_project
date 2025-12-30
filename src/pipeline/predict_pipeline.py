import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:

    def __init__(self):
        try:
            preproc_path = os.path.join("artifacts", "preprocessor.pkl")
            model_task1_path = os.path.join("artifacts", "model_task1.pkl")
            model_task2_path = os.path.join("artifacts", "model_task2.pkl")
            model_task3_path = os.path.join("artifacts", "model_task3.pkl")
            model_task4_path = os.path.join("artifacts", "model_task4.pkl")

            logging.info("Loading preprocessor and models from artifacts/ ...")

            self.preprocessor = load_object(preproc_path)
            self.model_task1 = load_object(model_task1_path)
            self.model_task2 = load_object(model_task2_path)
            self.model_task3 = load_object(model_task3_path)
            self.model_task4 = load_object(model_task4_path)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame) -> dict:
        try:
            logging.info("Starting prediction for %d rows", input_df.shape[0])

            X_scaled = self.preprocessor.transform(input_df)

            y1 = self.model_task1.predict(X_scaled)
            y2 = self.model_task2.predict(X_scaled)
            y3 = self.model_task3.predict(X_scaled)
            y4 = self.model_task4.predict(X_scaled)

            y1_val = int(y1[0])
            y2_val = float(y2[0])
            y3_val = int(y3[0])
            y4_val = int(y4[0])

            risk_map = {
                0: "An toàn",
                1: "Nguy cơ"
            }

            env_map = {
                0: "Môi trường không đạt",
                1: "Môi trường đạt"
            }

            algae_map = {
                0: "Điều kiện tảo kém",
                1: "Điều kiện tảo tốt"
            }

            result = {
                "task1_label": y1_val,
                "task1_text": risk_map.get(y1_val, "Không xác định"),

                "task2_vibrio_log": round(y2_val, 3),
                "task2_vibrio_est": float(np.round(np.exp(y2_val), 2)),

                "task3_label": y3_val,
                "task3_text": env_map.get(y3_val, "Không xác định"),

                "task4_label": y4_val,
                "task4_text": algae_map.get(y4_val, "Không xác định"),
            }

            return result

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        diem_quan_trac: str,
        xa: str,
        huyen: str,
        nhiet_do: float,
        ph: float,
        do: float,
        do_man: float,
        do_trong: float,
        do_kiem: float,
        no2: float,
        no3: float,
        nh4: float,
        po43: float,
        cod: float,
        nam: int,
        thang: int,
        ngay: int,
    ):
        self.diem_quan_trac = diem_quan_trac
        self.xa = xa
        self.huyen = huyen
        self.nhiet_do = nhiet_do
        self.ph = ph
        self.do = do
        self.do_man = do_man
        self.do_trong = do_trong
        self.do_kiem = do_kiem
        self.no2 = no2
        self.no3 = no3
        self.nh4 = nh4
        self.po43 = po43
        self.cod = cod
        self.nam = nam
        self.thang = thang
        self.ngay = ngay

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "DIEM_QUAN_TRAC": [self.diem_quan_trac],
                "XA": [self.xa],
                "HUYEN": [self.huyen],
                "NHIET_DO": [self.nhiet_do],
                "PH": [self.ph],
                "DO": [self.do],
                "DO_MAN": [self.do_man],
                "DO_TRONG": [self.do_trong],
                "DO_KIEM": [self.do_kiem],
                "NO2": [self.no2],
                "NO3": [self.no3],
                "NH4": [self.nh4],
                "PO43": [self.po43],
                "COD": [self.cod],
                "NAM": [self.nam],
                "THANG": [self.thang],
                "NGAY": [self.ngay],
            }

            df = pd.DataFrame(data)
            logging.info("CustomData converted to DataFrame with shape %s", df.shape)
            return df

        except Exception as e:
            raise CustomException(e, sys)
