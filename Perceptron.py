import numpy as np
import matplotlib.pyplot as plt


# Step 1: Generate a linearly separable dataset
def generate_linear_data(n_samples=100, margin=0.1):
    np.random.seed(0)
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.sign(X[:, 0] - X[:, 1] + margin)
    return X, y


# Step 2: Implement the simple perceptron class
class SimplePerceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.steps = 0

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            self.steps += 1
            errors = 0
            for xi, yi in zip(X, y):
                if yi * np.dot(self.weights, xi) <= 0:
                    self.weights += self.learning_rate * yi * xi
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return np.sign(np.dot(X, self.weights))

    def decision_function(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return np.dot(X, self.weights)


# Step 3: Visualize the dataset and decision boundary
def plot_decision_boundary(perceptron, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, levels=[-1, 0, 1], cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title('Decision Boundary')
    plt.show()


# Step 4: Test the perceptron
X, y = generate_linear_data(n_samples=100, margin=0.2)
perceptron = SimplePerceptron(learning_rate=0.1, max_iter=1000)
perceptron.fit(X, y)
plot_decision_boundary(perceptron, X, y)


# Step 5: Non-linear perceptron (homework task)
def generate_nonlinear_data(n_samples=100):
    np.random.seed(1)
    X = np.random.uniform([0, -1], [2 * np.pi, 1], (n_samples, 2))
    y = np.where(np.abs(np.sin(X[:, 0])) > np.abs(X[:, 1]), -1, 1)
    return X, y


def gaussian_kernel(X, centers, sigma):
    Z = np.exp(-np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2) / (2 * sigma ** 2))
    return Z


class NonLinearPerceptron:
    def __init__(self, learning_rate=0.01, max_iter=5000, n_centers=300, sigma=0.2):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_centers = n_centers
        self.sigma = sigma
        self.weights = None
        self.steps = 0

    def fit(self, X, y):
        self.centers = np.random.uniform(-1, 1, (self.n_centers, X.shape[1]))
        Z = gaussian_kernel(X, self.centers, self.sigma)
        Z = np.c_[np.ones(Z.shape[0]), Z]  # Add bias term
        self.weights = np.zeros(Z.shape[1])

        for _ in range(self.max_iter):
            self.steps += 1
            errors = 0
            for zi, yi in zip(Z, y):
                if yi * np.dot(self.weights, zi) <= 0:
                    self.weights += self.learning_rate * yi * zi
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        Z = gaussian_kernel(X, self.centers, self.sigma)
        Z = np.c_[np.ones(Z.shape[0]), Z]  # Add bias term
        return np.sign(np.dot(Z, self.weights))


# Step 6: Evaluate and visualize results
def evaluate_model(perceptron, X, y):
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Highlight misclassified points
    misclassified = X[y != y_pred]
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.scatter(misclassified[:, 0], misclassified[:, 1], facecolors='none', edgecolors='yellow', s=100,
                label='Misclassified')
    plt.legend()
    plt.title('Misclassified Points')
    plt.show()


# Step 7: Hyperparameter tuning
def tune_hyperparameters():
    accuracies = []
    sigmas = np.linspace(0.1, 0.3, 2)  # Zmniejszona liczba sigm
    centers_range = [100]  # Mniejsza liczba centrów

    for sigma in sigmas:
        for n_centers in centers_range:
            nonlinear_perceptron = NonLinearPerceptron(learning_rate=0.1, max_iter=500,  # Zmniejszone max_iter
                                                       n_centers=n_centers, sigma=sigma)
            nonlinear_perceptron.fit(X[:200], y[:200])  # Użycie próbki danych
            y_pred = nonlinear_perceptron.predict(X[200:])
            accuracy = np.mean(y_pred == y[200:])
            accuracies.append((sigma, n_centers, accuracy))

    best_params = max(accuracies, key=lambda x: x[2])
    print(f"Best parameters: sigma={best_params[0]}, n_centers={best_params[1]}, accuracy={best_params[2] * 100:.2f}%")




X, y = generate_nonlinear_data(n_samples=500)
nonlinear_perceptron = NonLinearPerceptron(learning_rate=0.1, max_iter=5000, n_centers=300, sigma=0.2)
nonlinear_perceptron.fit(X, y)
plot_decision_boundary(nonlinear_perceptron, X, y)
evaluate_model(nonlinear_perceptron, X, y)

# Optional: Uncomment to tune hyperparameters
tune_hyperparameters()
