import numpy as np
from typing import Tuple


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    return x, y

class Perceptron:
    def __init__(self, learning_rate: float, input_dim: int) -> None:
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.W = np.random.uniform(-1, 1, size=(self.input_dim, 1))

    def _update_weights(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        error = y - y_pred
        delta = error * x
        for d in delta:
            self.W = self.W + self.learning_rate * d.reshape(-1, 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        input_signal = np.dot(x, self.W)
        output_signal = (input_signal > 0.0).astype(np.int_) # step function
        return output_signal

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:

        def accuracy_score(y: np.ndarray, y_pred: np.ndarray) -> float:
            N = y.shape[0]
            accuracy = np.sum(y == y_pred) / N
            return float(accuracy)

        for epoch in range(1, epochs + 1):
            y_pred = self.predict(x)
            self._update_weights(x, y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            print(f"Epoch: {epoch} Accuracy: {accuracy}")


if __name__ == "__main__":
    x, y = get_dataset()
    p = Perceptron(learning_rate=0.05, input_dim=x.shape[1])
    p.train(x, y, 15)

    print(p.predict([1, 1]))