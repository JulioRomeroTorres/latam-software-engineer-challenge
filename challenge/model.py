import pandas as pd

import numpy as np
import datetime
from typing import Tuple, Union, List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

DEFAULT_THRESHOLD_MINUTES = 15

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
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
        data['delay'] = np.where(data['min_diff'] > self.threshold_in_minutes, 1, 0)

        return features,data[target_column]

    def determinate_weight_class(target: pd.Series)-> Dict[int, int]:
        total_len = len(target)
        mapper_class_weight = {}

        for class_value in target.unique().tolist():
            mapper_class_weight[class_value] = len(target[target == class_value])/total_len

        return mapper_class_weight

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

        x_train, _, y_train, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)

        class_weight = self.determinate_weight_class(y_train)
        model = LogisticRegression(class_weight=class_weight)

        model.fit(x_train, y_train)
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

        prediction_data = self._model.predict(features).tolist()
        return prediction_data