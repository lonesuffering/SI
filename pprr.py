import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import time
import copy

class MLPApproximator(BaseEstimator, RegressorMixin):

    ALGO_NAMES = ["sgd_simple", "sgd_momentum", "rmsprop", "adam"]

    def __init__(self, structure=[16, 8, 4], activation_name="relu", targets_activation_name="linear", initialization_name="uniform",
                 algo_name="sgd_simple", learning_rate=1e-2,  n_epochs=100, batch_size=10, seed=0,
                 verbosity_e=100, verbosity_b=10):
        self.structure = structure
        self.activation_name = activation_name
        self.targets_activation_name = targets_activation_name
        self.initialization_name = initialization_name
        self.algo_name = algo_name
        if self.algo_name not in self.ALGO_NAMES:
            self.algo_name = self.ALGO_NAMES[0]
        self.loss_name = "squared_loss"
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbosity_e = verbosity_e
        self.verbosity_b = verbosity_b

        self._initialize_weights()

    def _initialize_weights(self):
        np.random.seed(self.seed)
        self.weights = []
        self.biases = []

        layer_structure = [X.shape[1]] + self.structure

        for i in range(len(layer_structure) - 1):
            # Xavier initialization
            weight = np.random.randn(layer_structure[i], layer_structure[i + 1]) * np.sqrt(
                2. / (layer_structure[i] + layer_structure[i + 1]))
            bias = np.zeros((1, layer_structure[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_d(x):
        sig = MLPApproximator.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_d(x):
        return np.ones_like(x)

    @staticmethod
    def squared_loss(y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def squared_loss_d(y_true, y_pred):
        return y_pred - y_true

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            self.z_values.append(z)
            activation = self.sigmoid(z) if self.activation_name == "sigmoid" else self.linear(z)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        delta = self.squared_loss_d(y, self.activations[-1])
        for i in reversed(range(len(self.weights))):
            gradients_w[i] = np.dot(self.activations[i].T, delta) / m
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * (
                    self.sigmoid_d(self.z_values[i - 1]) if self.activation_name == "sigmoid" else self.linear_d(self.z_values[i - 1])
                )

        return gradients_w, gradients_b

    def algo_sgd_simple(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def fit(self, X, y):
        print(f"Starting training with {len(X)} samples...")
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            for batch_start in range(0, X.shape[0], self.batch_size):
                X_b = X[batch_start:batch_start + self.batch_size]
                y_b = y[batch_start:batch_start + self.batch_size]

                self.forward(X_b)
                gradients_w, gradients_b = self.backward(X_b, y_b)

                if self.algo_name == "sgd_simple":
                    self.algo_sgd_simple(gradients_w, gradients_b)

                batch_loss = self.squared_loss(y_b, self.activations[-1])
                epoch_loss += batch_loss

            epoch_loss /= (X.shape[0] // self.batch_size)
            if epoch % self.verbosity_e == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {epoch_loss:.4f}")

        print("Training completed.")

    def predict(self, X):
        print(f"Making predictions for {len(X)} samples...")
        return self.forward(X)

if __name__ == "__main__":
    def fake_data(n_samples=100, n_features=16, noise=0.1):
        X = np.random.rand(n_samples, n_features)
        y = np.sum(X, axis=1, keepdims=True) + noise * np.random.randn(n_samples, 1)
        return X, y

    # Настройки
    structure = [32, 16, 8, 1]  # Увеличение количества нейронов в слоях
    n_epochs = 2000             # Увеличение числа эпох
    batch_size = 10
    learning_rate = 0.1         # Увеличение скорости обучения

    # Генерация данных
    X, y = fake_data(n_samples=200, n_features=16)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Нормализация данных
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    # Создание и обучение модели
    model = MLPApproximator(
        structure=structure,
        activation_name="sigmoid",
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbosity_e=100
    )

    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    loss = model.squared_loss(y_test, y_pred)
    print(f"Test Loss: {loss:.4f}")
