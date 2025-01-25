import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.base import BaseEstimator, RegressorMixin
import time
import copy


class MLPApproximator(BaseEstimator, RegressorMixin):
    ALGO_NAMES = ["sgd_simple", "sgd_momentum", "rmsprop", "adam"]

    def __init__(self, structure=[32, 16, 8], activation_name="relu", targets_activation_name="linear",
                 initialization_name="uniform",
                 algo_name="sgd_simple", learning_rate=1e-2, n_epochs=100, batch_size=10, seed=0,
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
        self.history_weights = {}
        self.history_weights0 = {}
        self.n_params = None
        # params / constants for algorithms
        self.momentum_beta = 0.9
        self.rmsprop_beta = 0.9
        self.rmsprop_epsilon = 1e-7
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-7
        self.gradients = [None] * (len(self.structure) + 1)
        self.gradients0 = [None] * (len(self.structure) + 1)



    def __str__(self):
        txt = f"{self.__class__.__name__}(structure={self.structure},"
        txt += "\n" if len(self.structure) > 32 else " "
        txt += f"activation_name={self.activation_name}, targets_activation_name={self.targets_activation_name}, initialization_name={self.initialization_name}, "
        txt += f"algo_name={self.algo_name}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}, batch_size={self.batch_size})"
        if self.n_params:
            txt += f" [n_params: {self.n_params}]"
        return txt

    @staticmethod
    def he_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / n_in)
        return ((np.random.rand(n_out, n_in) * 2.0 - 1.0) * scaler).astype(np.float32)

    @staticmethod
    def he_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / n_in)
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def glorot_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / (n_in + n_out))
        return ((np.random.rand(n_out, n_in) * 2.0 - 1.0) * scaler).astype(np.float32)

    @staticmethod
    def glorot_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def prepare_batch_ranges(m, batch_size):
        n_batches = int(np.ceil(m / batch_size))
        batch_ranges = batch_size * np.ones(n_batches, dtype=np.int32)
        remainder = m % batch_size
        if remainder > 0:
            batch_ranges[-1] = remainder
        batch_ranges = np.r_[0, np.cumsum(batch_ranges)]
        return n_batches, batch_ranges

    @staticmethod
    def sigmoid(S):
        return 1 / (1 + np.exp(-S))

    @staticmethod
    def sigmoid_d(phi_S):
        return phi_S * (1 - phi_S)

    @staticmethod
    def relu(S):
        return np.maximum(0, S)

    @staticmethod
    def relu_d(phi_S):
        return (phi_S > 0).astype(float)

    @staticmethod
    def linear(S):
        return S

    @staticmethod
    def linear_d(phi_S):
        return np.ones_like(phi_S)

    @staticmethod
    def squared_loss(y_MLP, y_target):
        return 0.5 * (y_MLP - y_target) ** 2

    @staticmethod
    def squared_loss_d(y_MLP, y_target):
        return y_MLP - y_target

    def pre_algo_sgd_simple(self):
        return  # no special preparation needed for simple SGD



    def algo_sgd_simple(self, l):
        # Ensure gradients for weights and biases are initialized
        if not hasattr(self, 'gradients'):
            self.gradients = [None] * (len(self.structure) + 1)
        if not hasattr(self, 'gradients0'):
            self.gradients0 = [None] * (len(self.structure) + 1)
        # Проверка корректности градиентов после инициализации
        if self.gradients[l] is None or self.gradients0[l] is None:
            raise ValueError(f"Gradients for layer {l} are not computed correctly.")

        # Отладочный вывод
        if self.verbosity_e > 0:
            print(f"Layer {l}:")
            print(f"  Gradients shape: {self.gradients[l].shape}")
            print(f"  Bias gradients shape: {self.gradients0[l].shape}")
            print(f"  Weights shape: {self.weights_[l].shape}")
            print(f"  Bias weights shape: {self.weights0_[l].shape}")

        # Проверка совпадения размерностей
        if self.gradients[l].shape != self.weights_[l].shape:
            raise ValueError(f"Shape mismatch for layer {l} gradients and weights: "
                             f"gradients {self.gradients[l].shape}, weights {self.weights_[l].shape}.")
        if self.gradients0[l].shape != self.weights0_[l].shape:
            raise ValueError(f"Shape mismatch for layer {l} bias gradients and bias weights: "
                             f"gradients {self.gradients0[l].shape}, weights {self.weights0_[l].shape}.")

        # Отладочный вывод перед обновлением весов
        print(f"Updating Layer {l}:")
        print(f"  Gradients: {self.gradients[l]}")
        print(f"  Bias Gradients: {self.gradients0[l]}")
        print(f"  Weights Before Update: {self.weights_[l]}")
        print(f"  Biases Before Update: {self.weights0_[l]}")

        # Обновление весов
        self.weights_[l] -= self.learning_rate * self.gradients[l]
        self.weights0_[l] -= self.learning_rate * self.gradients0[l]

        # Отладочный вывод после обновления весов
        print(f"  Weights After Update: {self.weights_[l]}")
        print(f"  Biases After Update: {self.weights0_[l]}")

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.activation_ = getattr(MLPApproximator, self.activation_name)
        self.activation_d_ = getattr(MLPApproximator, self.activation_name + "_d")
        self.initialization_ = getattr(MLPApproximator,
                                       (
                                           "he_" if self.activation_name == "relu" else "glorot_") + self.initialization_name)
        self.targets_activation_ = getattr(MLPApproximator, self.targets_activation_name)
        self.targets_activation_d_ = getattr(MLPApproximator, self.targets_activation_name + "_d")
        self.loss_ = getattr(MLPApproximator, self.loss_name)
        self.loss_d_ = getattr(MLPApproximator, self.loss_name + "_d")
        self.pre_algo_ = getattr(self, "pre_algo_" + self.algo_name)
        self.algo_ = getattr(self, "algo_" + self.algo_name)

        # Начальная инициализация weights_ и weights0_
        self.weights_ = [None]
        self.weights0_ = [None]

        m, n = X.shape
        if len(y.shape) == 1:
            y = np.array([y]).T
        self.n_ = n
        self.n_targets_ = 1 if len(y.shape) == 1 else y.shape[1]
        self.n_params = 0

        for l in range(len(self.structure) + 1):
            n_in = n if l == 0 else self.structure[l - 1]
            n_out = self.structure[l] if l < len(self.structure) else self.n_targets_

            # Создание и добавление весов и смещений
            w = self.initialization_(n_in, n_out)
            w0 = np.zeros((n_out, 1), dtype=np.float32)
            self.weights_.append(w)
            self.weights0_.append(w0)

            # Обновление количества параметров
            self.n_params += w.size
            self.n_params += w0.size

            # Теперь можно проводить проверки
            assert self.weights_[-1].shape == (n_out, n_in), \
                f"Ошибка в инициализации весов слоя {l}! Ожидаемая форма: {(n_out, n_in)}, но получили {self.weights_[-1].shape}"
            assert self.weights0_[-1].shape == (n_out, 1), \
                f"Ошибка в инициализации смещений слоя {l}! Ожидаемая форма: {(n_out, 1)}, но получили {self.weights0_[-1].shape}"

        t1 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT [total of weights (params): {self.n_params}]")

        self.pre_algo_()  # подготовка алгоритма

        n_batches, batch_ranges = MLPApproximator.prepare_batch_ranges(m, self.batch_size)
        self.t = 0

        for e in range(self.n_epochs):
            t1_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                print("---")
                print(f"EPOCH {e + 1}/{self.n_epochs}:")
                self.forward(X)
                loss_e_before = np.mean(self.loss_(self.signals[-1], y))
            p = np.random.permutation(m)
            for b in range(n_batches):
                indexes = p[batch_ranges[b]: batch_ranges[b + 1]]
                X_b = X[indexes]
                y_b = y[indexes]
                self.forward(X_b)
                loss_b_before = np.mean(self.loss_(self.signals[-1], y_b))
                self.backward(y_b)
                for l in range(1, len(self.structure) + 2):
                    self.algo_(l)
                if (e % self.verbosity_e == 0 or e == self.n_epochs - 1) and b % self.verbosity_b == 0:
                    self.forward(X_b)
                    loss_b_after = np.mean(self.loss_(self.signals[-1], y_b))
                    print(f"[epoch {e + 1}/{self.n_epochs}, batch {b + 1}/{n_batches} -> "
                          f"loss before: {loss_b_before}, loss after: {loss_b_after}]")
                self.t += 1
            t2_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                self.forward(X)
                loss_e_after = np.mean(self.loss_(self.signals[-1], y))
                self.history_weights[e] = copy.deepcopy(self.weights_)
                self.history_weights0[e] = copy.deepcopy(self.weights0_)
                print(
                    f"ENDING EPOCH {e + 1}/{self.n_epochs} [loss before: {loss_e_before}, loss after: {loss_e_after}; epoch time: {t2_e - t1_e} s]")
        t2 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT DONE. [time: {t2 - t1} s]")
        if n != self.structure[0]:
            raise ValueError(
                f"Неправильная форма входных данных: ожидаем {self.structure[0]} признаков, но получили {n}.")

    def forward(self, X_b):
        self.signals = [None] * (len(self.structure) + 2)
        self.signals[0] = X_b
        for l in range(1, len(self.structure) + 2):
            S = np.dot(self.signals[l - 1], self.weights_[l].T) + self.weights0_[l].T
            self.signals[l] = self.activation_(S) if l <= len(self.structure) else self.targets_activation_(S)

    def backward(self, y):
        L = len(self.weights_) - 1  # Число слоев (включая выходной слой)
        self.deltas = [None] * (L + 1)

        # Дельта выходного слоя
        self.deltas[L] = (self.signals[L] - y) * self.activation_d_(
            self.signals[L])  # Заменено: signals_activated -> signals

        # Обратное распространение для скрытых слоев
        for l in range(L - 1, 0, -1):
            weights_transpose = self.weights_[l + 1].T  # Транспонирование весов следующего слоя

            # Проверка форм перед операциями
            assert self.deltas[l + 1].shape[1] == weights_transpose.shape[0], \
                f"Размеры не сопоставлены: deltas[{l + 1}].shape = {self.deltas[l + 1].shape}, weights_transpose.shape={weights_transpose.shape}"

            # Вычисление дельты
            self.deltas[l] = np.dot(self.deltas[l + 1], weights_transpose) * self.activation_d_(
                self.signals[l])  # Заменено: signals_activated -> signals

    def predict(self, X):
        self.forward(X)
        y_pred = self.signals[-1]
        if self.n_targets_ == 1:
            y_pred = y_pred[:, 0]
        return y_pred


def fake_data(m, domain=np.pi, noise_std=0.1):
    np.random.seed(0)
    X = np.random.rand(m, 2) * domain
    y = np.cos(X[:, 0] * X[:, 1]) * np.cos(2 * X[:, 0]) + np.random.randn(
        m) * noise_std  # target: cos(x_1 * x_2) * cos(2 * x_1) + normal noise
    return X, y


def loss_during_fit(approx, X_train, y_train, X_test, y_test):
    keys = list(approx.history_weights.keys())
    epochs = []
    losses_train = []
    losses_test = []
    weights = approx.weights_
    weights0 = approx.weights0_
    for k in keys:
        epochs.append(k + 1)
        approx.weights_ = approx.history_weights[k]
        approx.weights0_ = approx.history_weights0[k]
        losses_train.append(np.mean((approx.predict(X_train) - y_train) ** 2))
        losses_test.append(np.mean((approx.predict(X_test) - y_test) ** 2))
    approx.weights_ = weights
    approx.weights0_ = weights0
    return epochs, losses_train, losses_test


def r2_during_fit(approx, X_train, y_train, X_test, y_test):
    keys = list(approx.history_weights.keys())
    epochs = []
    r2s_train = []
    r2s_test = []
    weights = approx.weights_
    weights0 = approx.weights0_
    for k in keys:
        epochs.append(k + 1)
        approx.weights_ = approx.history_weights[k]
        approx.weights0_ = approx.history_weights0[k]
        r2s_train.append(approx.score(X_train, y_train))
        r2s_test.append(approx.score(X_test, y_test))
    approx.weights_ = weights
    approx.weights0_ = weights0
    return epochs, r2s_train, r2s_test


def initialize_weights(layers, init_type='uniform'):
    """
    Инициализация весов и смещений (bias) для нейронной сети.

    layers: список целых чисел, где каждый элемент — это количество нейронов в соответствующем слое.
    init_type: тип инициализации весов. Возможные варианты: 'uniform', 'he', 'glorot'.

    Возвращает два списка:
    - weights: список весовых матриц для каждого слоя;
    - biases: список смещений (bias) для каждого слоя.
    """
    weights = []
    biases = []

    for i in range(len(layers) - 1):
        n_in = layers[i]  # Количество нейронов во входящем слое
        n_out = layers[i + 1]  # Количество нейронов в текущем (выходящем) слое

        if init_type == 'uniform':
            # Инициализация в диапазоне [-0.5, 0.5]
            weight_matrix = np.random.uniform(-0.5, 0.5, (n_in, n_out))
        elif init_type == 'he':
            # Инициализация весов по Хе для активации ReLU
            weight_matrix = np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
        elif init_type == 'glorot':
            # Инициализация весов по Глороту для сигмоидальной функции
            limit = np.sqrt(6. / (n_in + n_out))
            weight_matrix = np.random.uniform(-limit, limit, (n_in, n_out))
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        # Создаем смещения (bias) в виде вектора с нулями
        bias_vector = np.zeros((1, n_out))

        # Добавляем веса и смещения в соответствующие списки
        weights.append(weight_matrix)
        biases.append(bias_vector)

        # Отладочная информация
        print(f"Layer {i + 1}: Weights initialized with shape {weight_matrix.shape}")
        print(f"Layer {i + 1}: Biases initialized with shape {bias_vector.shape}")

    return weights, biases

if __name__ == '__main__':
    print("MLP DEMO...")

    # DATA
    domain = 1.0 * np.pi
    noise_std = 0.1
    m_train = 1000
    m_test = 10000
    data_settings_str = f"{domain=}, {noise_std=}, {m_train=}, {m_test=}"
    print(f"DATA SETTINGS: {data_settings_str}")
    X_train, y_train = fake_data(m_train, domain, noise_std)
    X_test, y_test = fake_data(m_test, domain, noise_std)

    # Проверка инициализации весов отдельно
    layers = [2, 32, 16, 8, 1]  # Для создания сети: 2 входных, три скрытых (32, 16, 8), 1 выходной
    init_type = "uniform"  # Можно заменить на "he" или "glorot"

    print("Testing weight and bias initialization...")
    weights, biases = initialize_weights(layers, init_type=init_type)

    # Проверка результатов инициализации
    for i in range(len(weights)):
        print(f"Layer {i + 1}: Weights shape: {weights[i].shape} | Biases shape: {biases[i].shape}")

    print(f"\nWeights and biases initialized for {len(weights)} layers.\n")
    input_size = X_train.shape[1]  # Количество столбцов в X_train
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1  # Поддержка многомерного или скалярного y_train
    # APPROXIMATOR (NEURAL NETWORK)
    approx = MLPApproximator(
        structure=[2, 32, 16, 8, 1],  # Нужно включить входной (2) и выходной слой (1)
        activation_name="sigmoid",
        targets_activation_name="linear",
        initialization_name="uniform",
        algo_name="sgd_simple",
        learning_rate=1e-2,
        n_epochs=100,
        batch_size=10,
        seed=0,
        verbosity_e=100,
        verbosity_b=10
    )
    print(f"APPROXIMATOR (NEURAL NETWORK): {approx}")

    # FIT
    approx.fit(X_train, y_train)

    # METRICS - LOSS, R^2
    y_pred = approx.predict(X_train)
    mse = np.mean((y_pred - y_train) ** 2)
    print(f"LOSS TRAIN (MSE): {mse}")
    y_pred_test = approx.predict(X_test)
    mse_test = np.mean((y_pred_test - y_test) ** 2)
    print(f"LOSS TEST (MSE): {mse_test}")
    print(f"R^2 TRAIN: {approx.score(X_train, y_train)}")
    print(f"R^2 TEST: {approx.score(X_test, y_test)}")
    print("MLP DEMO DONE.")

    # PLOTS
    mesh_size = 50
    X1, X2 = np.meshgrid(np.linspace(0.0, domain, mesh_size), np.linspace(0.0, domain, mesh_size))
    X12 = np.array([X1.ravel(), X2.ravel()]).T
    y_ref = np.cos(X12[:, 0] * X12[:, 1]) * np.cos(2 * X12[:, 0])
    Y_ref = np.reshape(y_ref, (mesh_size, mesh_size))
    y_pred = approx.predict(X12)
    Y_pred = np.reshape(y_pred, (mesh_size, mesh_size))
    epochs, losses_train, losses_test = loss_during_fit(approx, X_train, y_train, X_test, y_test)
    epochs, r2s_train, r2s_test = r2_during_fit(approx, X_train, y_train, X_test, y_test)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"DATA SETTINGS: {data_settings_str}\nAPPROXIMATOR (NEURAL NETWORK): {approx}", fontsize=8)
    ax_loss = fig.add_subplot(2, 2, 1)
    ax_loss.set_title("TRAIN / TEST LOSS DURING FIT (MSE - MEAN SQUARED ERROR)")
    ax_loss.plot(epochs, losses_train, color="blue", marker=".", label="LOSS ON TRAIN DATA")
    ax_loss.plot(epochs, losses_test, color="red", marker=".", label="LOSS ON TEST DATA")
    ax_loss.legend()
    ax_loss.grid(color="gray", zorder=0, dashes=(4.0, 4.0))
    ax_loss.set_xlabel("EPOCH")
    ax_loss.set_ylabel("SQUARED LOSS")
    ax_r2 = fig.add_subplot(2, 2, 2)
    ax_r2.set_title("TRAIN / TEST $R^2$ DURING FIT (COEF. OF DETERMINATION)")
    ax_r2.plot(epochs, r2s_train, color="blue", marker=".", label="$R^2$ ON TRAIN DATA")
    ax_r2.plot(epochs, r2s_test, color="red", marker=".", label="$R^2$ ON TEST DATA")
    ax_r2.set_ylim(-0.25, 1.05)
    ax_r2.legend()
    ax_r2.grid(color="gray", zorder=0, dashes=(4.0, 4.0))
    ax_r2.set_xlabel("EPOCH")
    ax_r2.set_ylabel("$R^2$")
    ax_train_data = fig.add_subplot(2, 3, 4, projection='3d')
    ax_target = fig.add_subplot(2, 3, 5, projection='3d')
    ax_approximator = fig.add_subplot(2, 3, 6, projection='3d')
    ax_train_data.set_title("TRAINING DATA", pad=-32)
    ax_train_data.scatter(X_train[:, 0], X_train[:, 1], y_train, marker=".")
    ax_target.set_title("TARGET (TO BE APPROXIMATED)", pad=-128)
    ax_target.plot_surface(X1, X2, Y_ref, cmap=cm.get_cmap("Spectral"))
    ax_approximator.set_title("NEURAL APPROXIMATOR")
    ax_approximator.plot_surface(X1, X2, Y_pred, cmap=cm.get_cmap("Spectral"))
    ax_train_data.set_box_aspect([2, 2, 1])
    ax_target.set_box_aspect([2, 2, 1])
    ax_approximator.set_box_aspect([2, 2, 1])
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.9, hspace=0.25, wspace=0.15)
    plt.show()
