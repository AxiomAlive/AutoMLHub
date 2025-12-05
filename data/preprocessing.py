import logging
from typing import Tuple, Union, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from loguru import logger


class CategoricalFeaturePreprocessor:
    def encode(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
        if isinstance(X, pd.DataFrame):
            X.dropna(inplace=True)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        if isinstance(y, pd.Series):
            y_encoded = pd.Series(y_encoded)

        if isinstance(X, pd.DataFrame):
            X_encoded = X.copy(deep=True)
            for dataset_feature_name in X:
                dataset_feature = X.get(dataset_feature_name)

                if type(dataset_feature.iloc[0]) is str:
                    dataset_feature_encoded = pd.get_dummies(dataset_feature, prefix=dataset_feature_name)
                    X_encoded.drop([dataset_feature_name], axis=1, inplace=True)
                    X_encoded = pd.concat([X_encoded, dataset_feature_encoded], axis=1)
                    X_encoded.reset_index(drop=True, inplace=True)

            assert len(X_encoded.index) == len(y_encoded.index), f"X index size is {len(X_encoded.index)} and y index size is {len(y_encoded.index)}."
        else:
            X_encoded = X

        return X_encoded, y_encoded
