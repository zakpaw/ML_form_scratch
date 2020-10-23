import numpy as np


def distance(a: int, b: int) -> float:
    """[Calculate l2 norm which is Euclidean distance
        between a and b]

    Args:
        a (int): [1st point]
        b (int): [2nd point]

    Returns:
        float: [Distance between a and b]
    """
    return np.linalg.norm(a-b)


class KNearestNeighbors(object):
    def __init__(self, k: int = 5):
        """[Initialize KNN class with k]

        Args:
            k (int, optional): [number of nearest neighbors]. Defaults to 5.
        """
        self.X, self.Y = None, None
        self.classes = None
        self.k = k

    def fit(self, X: np.array, Y: np.array) -> None:
        """[Load data into RAM]

        Args:
            X (np.array): [X_train data]
            Y (np.array): [Y_train features]
        """
        self.X, self.Y = X, Y

    def predict(self, new_X: np.array) -> np.array:
        """[Predict the class label of given points]

        Args:
            new_X (np.array): [X_test data]

        Returns:
            np.array: [Y_test features]
        """
        Y_pred = np.zeros(len(new_X))
        for i, new in enumerate(new_X):
            dist_neighbors = []
            for x, y in zip(self.X, self.Y):
                eucl_d = distance(new, x)
                dist_neighbors.append([eucl_d, y])
            # sort ascending based on distances
            dist_neighbors = sorted(dist_neighbors, 
                                    key=lambda x: x[0])[:self.k]
            # extract 1st column from each row
            col1 = lambda x: [x[i][1] for i, _ in enumerate(x)]
            # find the most common label
            Y_pred[i] = np.bincount(col1(dist_neighbors)).argmax()
        return Y_pred

    def score(self, y_test: np.array, y_pred: np.array) -> float:
        """[Calculate accuracy score]

        Args:
            y_test (np.array): [correct labels]
            y_pred (np.array): [predicted labels]

        Returns:
            float: [accuarcy store, 1=100%, 0=0%]
        """
        return float(np.sum(y_pred == y_test))/len(y_test)


# Check if it works
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

df = load_iris(as_frame=True).frame.values
X, y = df[:, :-1], df[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,
                                                    random_state=420)
model = KNearestNeighbors(k=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(y_test, y_pred))
