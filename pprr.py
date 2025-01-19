import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fake_data(n_samples=200, n_features=16, noise=0.1, domain=2 * np.pi):
    """
    Generates synthetic data for regression tasks.
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.uniform(0, domain, (n_samples, n_features))
    y = np.sum(X, axis=1, keepdims=True) + noise * np.random.randn(n_samples, 1)
    return X, y

# Настройки
domain = 1.5 * np.pi  # Изменение размера домена
n_samples = 200
n_features = 16
noise = 0.1

# Генерация данных
X, y = fake_data(n_samples=n_samples, n_features=n_features, noise=noise, domain=domain)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Нормализация данных
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std  # Используем среднее и стандартное отклонение обучающих данных

# Применение PCA (опционально)
# pca = PCA(n_components=0.95)  # Retain 95% of the variance
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# Создание и обучение модели
class MLPApproximator(BaseEstimator, RegressorMixin):

    ALGO_NAMES = ["sgd_simple", "sgd_momentum", "rmsprop", "adam"]

    def __init__(self, structure=[16, 8, 4], activation_name="relu", targets_activation_name="linear", initialization_name="uniform",
                 algo_name="sgd_simple", learning_rate=1e-2,  n_epochs=100, batch_size=10, seed=0,
                 verbosity_e=100, verbosity_b=10, regularization=None, reg_lambda=0.01):
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
        self.regularization = regularization
        self.reg_lambda = reg_lambda

        self._initialize_weights()
        self._initialize_algorithms()

    def _initialize_weights(self):
        np.random.seed(self.seed)
        self.weights = []
        self.biases = []

        # Include input layer in layer_structure
        layer_structure = [self.structure[0]] + self.structure[1:]

        for i in range(len(layer_structure) - 1):
            input_features = layer_structure[i]
            output_features = layer_structure[i + 1]
            # Initialize weights with shape (input_features, output_features)
            weight = np.random.uniform(-1, 1, (input_features, output_features))
            # Initialize biases with shape (1, output_features)
            bias = np.random.uniform(-1, 1, (1, output_features))
            self.weights.append(weight)
            self.biases.append(bias)

    def _initialize_algorithms(self):
        if self.algo_name == "sgd_momentum":
            self.pre_algo_momentum()
        elif self.algo_name == "rmsprop":
            self.pre_algo_rmsprop()
        elif self.algo_name == "adam":
            self.pre_algo_adam()

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_d(x):
        return np.where(x > 0, 1, 0)

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

        for i in range(len(self.weights)):
            w, b = self.weights[i], self.biases[i]
            z = np.dot(self.activations[-1], w) + b
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                activation = self.relu(z) if self.activation_name == "relu" else (
                    self.sigmoid(z) if self.activation_name == "sigmoid" else self.linear(z))
            else:
                # Use linear activation for the output layer
                activation = self.linear(z)
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
                    self.relu_d(self.z_values[i - 1]) if self.activation_name == "relu" else (
                        self.sigmoid_d(self.z_values[i - 1]) if self.activation_name == "sigmoid" else self.linear_d(self.z_values[i - 1])
                    )
                )
            if self.regularization == 'l2':
                gradients_w[i] += self.reg_lambda * self.weights[i]
            elif self.regularization == 'l1':
                gradients_w[i] += self.reg_lambda * np.sign(self.weights[i])

        return gradients_w, gradients_b

    def algo_sgd_simple(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def pre_algo_momentum(self):
        self.momentum_v_w = [np.zeros_like(w) for w in self.weights]
        self.momentum_v_b = [np.zeros_like(b) for b in self.biases]
        self.momentum_beta = 0.9

    def algo_momentum(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.momentum_v_w[i] = self.momentum_beta * self.momentum_v_w[i] + (1 - self.momentum_beta) * gradients_w[i]
            self.momentum_v_b[i] = self.momentum_beta * self.momentum_v_b[i] + (1 - self.momentum_beta) * gradients_b[i]
            self.weights[i] -= self.learning_rate * self.momentum_v_w[i]
            self.biases[i] -= self.learning_rate * self.momentum_v_b[i]

    def pre_algo_rmsprop(self):
        self.rms_s_w = [np.zeros_like(w) for w in self.weights]
        self.rms_s_b = [np.zeros_like(b) for b in self.biases]
        self.rms_beta = 0.9
        self.epsilon = 1e-8

    def algo_rmsprop(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.rms_s_w[i] = self.rms_beta * self.rms_s_w[i] + (1 - self.rms_beta) * gradients_w[i] ** 2
            self.rms_s_b[i] = self.rms_beta * self.rms_s_b[i] + (1 - self.rms_beta) * gradients_b[i] ** 2
            self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.rms_s_w[i]) + self.epsilon)
            self.biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.rms_s_b[i]) + self.epsilon)

    def pre_algo_adam(self):
        self.adam_m_w = [np.zeros_like(w) for w in self.weights]
        self.adam_m_b = [np.zeros_like(b) for b in self.biases]
        self.adam_v_w = [np.zeros_like(w) for w in self.weights]
        self.adam_v_b = [np.zeros_like(b) for b in self.biases]
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def algo_adam(self, gradients_w, gradients_b):
        self.t += 1
        for i in range(len(self.weights)):
            self.adam_m_w[i] = self.adam_beta1 * self.adam_m_w[i] + (1 - self.adam_beta1) * gradients_w[i]
            self.adam_m_b[i] = self.adam_beta1 * self.adam_m_b[i] + (1 - self.adam_beta1) * gradients_b[i]
            self.adam_v_w[i] = self.adam_beta2 * self.adam_v_w[i] + (1 - self.adam_beta2) * gradients_w[i] ** 2
            self.adam_v_b[i] = self.adam_beta2 * self.adam_v_b[i] + (1 - self.adam_beta2) * gradients_b[i] ** 2

            m_w_hat = self.adam_m_w[i] / (1 - self.adam_beta1 ** self.t)
            m_b_hat = self.adam_m_b[i] / (1 - self.adam_beta1 ** self.t)
            v_w_hat = self.adam_v_w[i] / (1 - self.adam_beta2 ** self.t)
            v_b_hat = self.adam_v_b[i] / (1 - self.adam_beta2 ** self.t)

            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def fit(self, X, y):
        # Проверяем, чтобы размер входных данных X совпадал с размером первого слоя
        if X.shape[1] != self.structure[0]:
            raise ValueError(
                f"Размер входных данных ({X.shape[1]}) не соответствует размеру первого слоя сети ({self.structure[0]}).")

        logging.info(f"Starting training with {X.shape[0]} samples...")
        self.loss_history = []
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            for batch_start in range(0, X.shape[0], self.batch_size):
                X_b = X[batch_start:batch_start + self.batch_size]
                y_b = y[batch_start:batch_start + self.batch_size]

                self.forward(X_b)
                gradients_w, gradients_b = self.backward(X_b, y_b)

                if self.algo_name == "sgd_simple":
                    self.algo_sgd_simple(gradients_w, gradients_b)
                elif self.algo_name == "sgd_momentum":
                    self.algo_momentum(gradients_w, gradients_b)
                elif self.algo_name == "rmsprop":
                    self.algo_rmsprop(gradients_w, gradients_b)
                elif self.algo_name == "adam":
                    self.algo_adam(gradients_w, gradients_b)

                batch_loss = self.squared_loss(y_b, self.activations[-1])
                epoch_loss += batch_loss

            epoch_loss /= (X.shape[0] // self.batch_size)
            self.loss_history.append(epoch_loss)
            if epoch % self.verbosity_e == 0:
                logging.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    def predict(self, X):
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            activation = self.relu(z) if self.activation_name == "relu" else (
                self.sigmoid(z) if self.algo_name == "sigmoid" else self.linear(z))
            activations.append(activation)
        return activations[-1]

import matplotlib.pyplot as plt
import pickle
if __name__ == "__main__":
    logging.info("Training started...")

    # Training parameters
    structure = [X_train.shape[1], 16, 8, 1]
    n_epochs = 100
    batch_size = 10
    learning_rate = 1e-3
    activation_name = "relu"
    algo_name = "adam"
    seed = 53867

    # Data generation
    X, y = fake_data(n_samples=200, n_features=16, noise=noise, domain=domain)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Normalization
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

    # Creating the model
    model = MLPApproximator(
        structure=structure,
        activation_name=activation_name,
        algo_name=algo_name,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbosity_e=10,
        seed=seed,
    )

    # Training the model
    logging.info("Training the model...")
    model.fit(X_train, y_train)
    logging.info("Training completed.")

    # Plotting the training loss
    if hasattr(model, "loss_history") and len(model.loss_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_history, label="Training Loss")
        plt.legend()
        plt.title("Training Process")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
    else:
        logging.warning("Loss history (loss_history) is empty or missing.")

    # Performing predictions
    logging.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculating errors
    loss = model.squared_loss(y_test, y_pred)
    logging.info(f"Test Loss: {loss:.4f}")

    # Comparing true and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True Values", color="blue")
    plt.plot(y_pred, label="Predicted Values", color="orange")
    plt.legend()
    plt.title("Comparison of True and Predicted Values")
    plt.show()

    # Experimenting with different parameters
    logging.info("Starting experiments with various parameters...")
    structures = [
        [X_train.shape[1], 128, 64, 32],
        [X_train.shape[1], 128, 128, 64, 64, 32, 32],
        [X_train.shape[1]] + [64] * 5 + [32] * 5 + [16] * 5 + [8] * 5
    ]

    results = []

    for structure in structures:
        for activation in ["sigmoid", "relu"]:
            for algo in ["sgd_simple", "momentum", "rmsprop", "adam"]:
                for lr in [1e-2, 1e-3, 1e-4]:
                    logging.info(
                        f"Training model with structure={structure}, activation={activation}, algo={algo}, lr={lr}")
                    model = MLPApproximator(
                        structure=structure,
                        activation_name=activation,
                        algo_name=algo,
                        learning_rate=lr,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        verbosity_e=100,
                        seed=seed,
                    )
                    model.fit(X_train, y_train)

                    # Saving the result
                    test_loss = model.squared_loss(y_test, y_pred)
                    results.append({
                        "structure": structure,
                        "activation": activation,
                        "algo": algo,
                        "learning_rate": lr,
                        "test_loss": test_loss,
                        "loss_history": model.loss_history if hasattr(model, "loss_history") else []
                    })

    # Comparative loss graph
    plt.figure(figsize=(12, 8))
    for result in results:
        if len(result["loss_history"]) > 0:
            plt.plot(
                result["loss_history"],
                label=f"Struct={result['structure']}, Act={result['activation']}, Algo={result['algo']}, LR={result['learning_rate']}"
            )
    plt.legend()
    plt.title("Comparative Loss Graph for Different Structures and Parameters")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Saving the results
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    logging.info("Experiments completed. Results saved.")
