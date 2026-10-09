"""TabPFN regressor for the CYP challenge.

TabPFN-3.5 accepts at most 20,000 features and 1,000,000 rows, so wide
representations may need reducing before they fit.
"""

from typing import ClassVar, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from models.base import CYPModel


class TabPFN(CYPModel):
    """In-context tabular regressor; no gradient training happens in fit().

    Callers filter missing labels with models.utils.drop_nan_rows.
    """

    name: ClassVar[str] = "tabpfn"

    def __init__(
        self,
        n_estimators: int | Literal["auto"] = "auto",
        softmax_temperature: float | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        device: str = "auto",
        model_path: str = "auto",
        random_state: int = 42,
    ) -> None:
        from tabpfn import TabPFNRegressor

        self._model = Pipeline(
            [
                (
                    "regressor",
                    TabPFNRegressor(
                        n_estimators=n_estimators,
                        softmax_temperature=softmax_temperature,
                        ignore_pretraining_limits=ignore_pretraining_limits,
                        device=device,
                        model_path=model_path,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.set_params(pca__n_components=min(self.n_components, *X.shape))
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
