import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch


class LogRegPCA:
    def __init__(self, pca=True):
        self.pca = PCA() if pca else None
        self.model = LogisticRegression()

    def model_training(self, x, y):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.fit_transform(x)

        self.model.fit(x, y)

        acc = self.model.score(x, y)
        print('Accuracy on train:', round(acc, 3))

        return acc

    def model_predict(self, x):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.transform(x)

        y_pred = self.model.predict(x)
        return y_pred

    def model_testing(self, x, y):
        y_pred = self.model_predict(x)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1

    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs

class SimpleCatBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators  # Количество деревьев
        self.learning_rate = learning_rate  # Скорость обучения
        self.max_depth = max_depth  # Максимальная глубина дерева
        self.models = []  # Для хранения деревьев

    def fit(self, X, y):
        y_pred = np.full(y.shape, np.mean(y))
        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X).reshape(-1, 1)  # Приводим предсказание к нужной форме
            self.models.append(tree)
            mse = mean_squared_error(y, y_pred)
            print(f'Iteration {i+1}/{self.n_estimators}, MSE: {mse:.4f}')

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output for 2 classes (binary classification)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Using CrossEntropyLoss which expects raw logits

    # Save model
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    # Load model
    def load_model(self, path):
        self.load_state_dict(torch.load(path))