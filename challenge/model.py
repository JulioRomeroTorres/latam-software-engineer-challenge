import pandas as pd

import numpy as np
from datetime import datetime
from typing import Tuple, Union, List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import joblib

DEFAULT_THRESHOLD_MINUTES = 15

class DelayModel:

    def __init__(
        self
    ):
        self._model = None
        self.top_features_names = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self.threshold_in_minutes = DEFAULT_THRESHOLD_MINUTES

    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)

        features = features[self.top_features_names]

        if target_column is None:
            return features
        
        target = pd.DataFrame({
            target_column: np.where(data['min_diff'] > self.threshold_in_minutes, 1, 0)
        })

        return features, target

    def determinate_weight_class(self, target: pd.Series)-> Dict[int, int]:
        total_len = len(target)

        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])

        return {1: n_y0/total_len, 0: n_y1/total_len}

    def show_metric(data: pd.Series, predicted_data: pd.Series):
        print("Confusion Matrix")
        current_confusion_matrix = confusion_matrix(data, predicted_data)
        print(current_confusion_matrix)
        print("*"*10)
        print("Report")
        print(classification_report(data, predicted_data))

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y_train = target['delay']
        class_weight = self.determinate_weight_class(y_train)
        model = LogisticRegression(class_weight=class_weight)
        model.fit(features, y_train)

        joblib.dump(model, './challenge/artifacts/model.joblib')

        self._model = model

        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        if self._model is None:
            self._model = joblib.load(f"./challenge/artifacts/model.joblib")

        prediction_data = self._model.predict(features).tolist()
        return prediction_data