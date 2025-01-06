import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

def load_adult_data():
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv("adult.data", header=None, names=column_names, na_values=" ?", skipinitialspace=True)
    data.dropna(inplace=True)
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("category").cat.codes
    X = data.drop(columns=["income"]).values
    y = data["income"].values
    return X, y

X, y = load_adult_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def discretize_data(X, n_bins, feature_min=None, feature_max=None):
    if feature_min is None:
        feature_min = X.min(axis=0)
    if feature_max is None:
        feature_max = X.max(axis=0)
    bins = [np.linspace(feature_min[i], feature_max[i], n_bins + 1) for i in range(X.shape[1])]
    X_discretized = np.array([np.digitize(X[:, i], bins[i][:-1], right=False) for i in range(X.shape[1])]).T
    return np.clip(X_discretized, 0, n_bins - 1)

class NaiveBayesDiscreteSafe(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=10, laplace_smoothing=True):
        self.n_bins = n_bins
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_min_ = X.min(axis=0)
        self.feature_max_ = X.max(axis=0)
        X_discretized = discretize_data(X, self.n_bins, self.feature_min_, self.feature_max_)
        self.class_prior_ = np.log(np.bincount(y) / len(y))
        self.feature_conditional_ = {}
        for cls in self.classes_:
            X_cls = X_discretized[y == cls]
            self.feature_conditional_[cls] = []
            for feature_idx in range(X.shape[1]):
                values, counts = np.unique(X_cls[:, feature_idx], return_counts=True)
                feature_prob = np.zeros(self.n_bins)
                for value, count in zip(values, counts):
                    feature_prob[value] = count
                if self.laplace_smoothing:
                    feature_prob += 1
                feature_prob = np.log(feature_prob / (len(X_cls) + self.n_bins))
                self.feature_conditional_[cls].append(feature_prob)
        return self

    def predict_proba(self, X):
        X_discretized = discretize_data(X, self.n_bins, self.feature_min_, self.feature_max_)
        log_probas = []
        raw_probas = []
        for x in X_discretized:
            class_log_probas = []
            for cls in self.classes_:
                log_prior = self.class_prior_[cls]
                log_likelihood = sum([
                    self.feature_conditional_[cls][i][x[i]]
                    if 0 <= x[i] < self.n_bins else -np.inf
                    for i in range(len(x))
                ])
                class_log_probas.append(log_prior + log_likelihood)
            log_probas.append(class_log_probas)
            raw_probas.append(np.exp(class_log_probas))
        normalized_probas = [
            probs / sum(probs) for probs in raw_probas
        ]
        return np.array(normalized_probas), np.array(raw_probas)

    def predict(self, X):
        probas, _ = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

nb_safe = NaiveBayesDiscreteSafe(n_bins=10, laplace_smoothing=True)
nb_safe.fit(X_train, y_train)


normalized_probas, raw_probas = nb_safe.predict_proba(X_test)

for i in range(5):
    print(f"Пример {i+1}:")
    print("Сырые вероятности (до нормализации):", raw_probas[i])
    print("Нормализованные вероятности (сумма = 1):", normalized_probas[i])
    print("-" * 50)

test_predictions = nb_safe.predict(X_test)
test_accuracy_safe = accuracy_score(y_test, test_predictions)
print(f"Точность модели (Safe): {test_accuracy_safe:.2f}")

