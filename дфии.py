import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_nursery_data():
    columns = [
        "parents", "has_nurs", "form", "children", "housing",
        "finance", "social", "health", "class"
    ]
    data = pd.read_csv('nursery.data', header=None, names=columns)
    label_encoders = {col: LabelEncoder() for col in data.columns}
    for col in data.columns:
        data[col] = label_encoders[col].fit_transform(data[col])
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


X, y = load_nursery_data()

# Поделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Функция дискретизации
def discretize(data, bins, min_vals=None, max_vals=None):
    if min_vals is None:
        min_vals = np.min(data, axis=0)
    if max_vals is None:
        max_vals = np.max(data, axis=0)
    bin_width = (max_vals - min_vals) / bins
    discretized = np.floor((data - min_vals) / bin_width).astype(int)
    return np.clip(discretized, 0, bins - 1)


# Класс для классификатора наивного Байеса
class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bins=10, laplace_smoothing=True):
        self.bins = bins
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_count_ = X.shape[1]
        self.class_prior_ = np.zeros(len(self.classes_))
        self.feature_likelihood_ = {}

        self.min_vals_ = np.min(X, axis=0)
        self.max_vals_ = np.max(X, axis=0)
        discretized_X = discretize(X, self.bins, self.min_vals_, self.max_vals_)

        for cls in self.classes_:
            cls_idx = np.where(self.classes_ == cls)[0][0]
            cls_indices = np.where(y == cls)[0]
            self.class_prior_[cls_idx] = np.log((len(cls_indices) + 1) / (len(y) + len(self.classes_)))

            class_feature_counts = []
            for feature_idx in range(self.feature_count_):
                feature_values, counts = np.unique(discretized_X[cls_indices, feature_idx], return_counts=True)
                likelihood = np.zeros(self.bins)
                likelihood[feature_values] = counts
                if self.laplace_smoothing:
                    likelihood += 1
                likelihood = np.log(likelihood / np.sum(likelihood))
                class_feature_counts.append(likelihood)

            self.feature_likelihood_[cls_idx] = np.array(class_feature_counts)

    def predict_proba(self, X):
        discretized_X = discretize(X, self.bins, self.min_vals_, self.max_vals_)
        probabilities = []

        for x in discretized_X:
            class_probs = []
            for cls in self.classes_:
                cls_idx = np.where(self.classes_ == cls)[0][0]
                prob = self.class_prior_[cls_idx]
                for feature_idx in range(self.feature_count_):
                    prob += self.feature_likelihood_[cls_idx][feature_idx, x[feature_idx]]
                class_probs.append(prob)
            probabilities.append(np.exp(class_probs) / np.sum(np.exp(class_probs)))

        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


# Обучение и тестирование классификатора
classifier = NaiveBayesClassifier(bins=10, laplace_smoothing=True)
classifier.fit(X_train, y_train)

# Прогнозирование
train_preds = classifier.predict(X_train)
test_preds = classifier.predict(X_test)

# Оценка точности
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f'Точность на обучающем наборе: {train_accuracy:.2f}')
print(f'Точность на тестовом наборе: {test_accuracy:.2f}')

# Вывод логарифмических и нормализованных вероятностей для 5 примеров
for example_index in range(5):
    log_probs = classifier.predict_proba(X_test[example_index:example_index + 1])[0]
    print(f"\nWyniki dla wszystkich klas (włączając poprawkę La-Place’a) wynoszą {log_probs}")

    # Нормализация вероятностей
    normalized_probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
    print(f"Dzieląc każdy wynik przez sumę wszystkich uzyskamy prawdopodobieństwa, które wynoszą {normalized_probs}")
