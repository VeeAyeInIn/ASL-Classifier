import numpy as np
import pandas as pd


def _distance(angles1: np.ndarray, angles2: np.ndarray, p: int = 2) -> float:
    differences = np.abs(angles1 - angles2)
    summation = np.sum([value ** p for value in differences])
    return summation ** (1 / p)


class ASLModel:

    def __init__(self):
        self.X = []
        self.y = []

    def train(self, inputs: pd.DataFrame, labels: pd.DataFrame):
        for i in range(inputs.shape[0]):
            if i % 20 != 0:
                continue
            self.X.append(inputs.iloc[i])
            self.y.append(labels.iloc[i][0])

    def test(self, inputs: pd.DataFrame, labels: pd.DataFrame) -> float:
        pass

    def predict_(self, inputs: list[list[float]], k: int = 1) -> list[str]:

        results = []

        for input_ in inputs:
            distances = [_distance(np.array(input_), training) for training in self.X]
            results.append(self.y[np.argmin(distances)])

        return results

    def predict(self, inputs: list[list[float]], k: int = 1) -> list[str]:

        results = []

        for input_ in inputs:
            distances = [_distance(np.array(input_), training) for training in self.X]

            nearest = []
            for i in range(k):
                smallest = np.argmin(distances)
                nearest.append(smallest)
                distances[smallest] = distances[np.argmax(distances)]

            # predictions = np.array([self.y[j] for j in nearest])
            predictions = [self.y[j] for j in nearest]
            results.append(max(set(predictions), key=predictions.count))

            # results.append(self.y[np.argmin(distances)])
            # results.append(self.y[np.mode(nearest)])

        return results
