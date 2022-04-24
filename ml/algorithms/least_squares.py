import numpy as np

from ml.algorithms.linear_regression import ParamsInicialize, Linear

class LeastSquares(ParamsInicialize):

    def fit_train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Linear:
        
        l2_reg_matrix = np.eye(x.shape[1]) * self.l2_regulazation
        
        w = np.linalg.inv( (x.T @ x) + l2_reg_matrix ) @ x.T @ y
        
        return Linear(w)