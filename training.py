import numpy as np
import pandas as pd


def _distance(angles1: np.ndarray, angles2: np.ndarray, p: int = 2) -> float:

    """
    Gets the distance between two n-dimensional points.
    :param angles1: The first set of n angles.
    :param angles2: The second set of n angles.
    :param p: Degree of distance (e.g. L2) to calculate.
    :return: The scalar distance between the two sets of angles.
    """

    differences = np.abs(angles1 - angles2)
    summation = np.sum([value ** p for value in differences])
    return summation ** (1 / p)


class ASLModel:

    """
    A simple modification of KNN for 18 dimensions.
    """

    def __init__(self):

        """
        Initializes empty training data and training labels.
        """

        self.X = []
        self.y = []

    def train(self, inputs: pd.DataFrame, labels: pd.DataFrame, fold: int = 0, folds: int = 20):

        """
        Trains the model according to the provided data and labels.
        :param inputs: The pandas DataFrame for training data.
        :param labels: The pandas DataFrame for labels.
        :param fold: The specific fold to use.
        :param folds: The amount of folds to use.
        """

        for i in range(inputs.shape[0]):
            if (i + fold) % folds != 0:
                continue
            self.X.append(inputs.iloc[i])
            self.y.append(labels.iloc[i][0])

    def test(self, inputs: pd.DataFrame, labels: pd.DataFrame) -> float:
        pass

    def predict_(self, inputs: list[list[float]], k: int = 1) -> list[str]:

        """
        Obsolete prediction function for predicting the labels of provided data.
        :param inputs: The data to predict the labels for.
        :param k: The amount of neighbors to compare self to.
        :return: The predicted labels of the data.
        """

        results = []

        for input_ in inputs:
            distances = [_distance(np.array(input_), training) for training in self.X]
            results.append(self.y[np.argmin(distances)])

        return results

    def predict(self, inputs: list[list[float]], k: int = 1) -> list[str]:

        """
        Predicts the label of data based on the distance to k other data points.
        :param inputs: The data to predict the labels for.
        :param k: The amount of neighbors to compare self to.
        :return: The predicted labels of the data.
        """

        results = []

        for input_ in inputs:
            distances = [_distance(np.array(input_), training) for training in self.X]

            nearest = []
            for i in range(k):
                smallest = np.argmin(distances)
                nearest.append(smallest)
                distances[smallest] = distances[np.argmax(distances)]

            predictions = [self.y[j] for j in nearest]
            results.append(max(set(predictions), key=predictions.count))

        return results
