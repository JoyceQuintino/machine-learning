import numpy as np
from abc import ABC
from typing import Any

class Linear:

    w: np.ndarray

    def __init__(self, w: np.ndarray):
        self.w = w.reshape([-1, 1])

    def model(lenght: int, fill_value: float) -> 'Linear':
        w = np.full(shape=lenght,fill_value=fill_value).reshape([-1, 1])
        return Linear(w)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w

    def update(self, w: np.ndarray):
        self.w = w.reshape([-1, 1])

    def copy_model(self):
        return Linear(self.w.reshape([-1, 1]))

class ParamsInicialize(ABC):

    initial_w_values: float
    seed: int
    ephocs: int
    alpha: float
    l2: float
    predictions: bool

    def __init__(self, alpha=0.01, ephocs=100, initial_w_values=1, l2=0, predictions=False, seed=1234):
        self.l2 = l2
        self.seed = seed
        self.predictions = predictions
        self.initial_w_values = initial_w_values
        self.ephocs = ephocs
        self.alpha = alpha

        def fit_train(self, x: np.ndarray, y: np.ndarray, **nargs): pass