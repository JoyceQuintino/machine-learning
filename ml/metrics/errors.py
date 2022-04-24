import numpy as np

class EvaluationMetrics:

    def rmse(y_real: np.ndarray, y_predicted: np.ndarray) -> float:
        diff = (y_real - y_predicted) ** 2

        result = np.sqrt(diff.mean())

        return result



