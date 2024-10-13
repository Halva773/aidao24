import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


class FullyConnectedNN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr, epochs):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs

        # Инициализация весов
        self.weights1 = np.random.randn(self.input_size, self.hidden1_size)
        self.weights2 = np.random.randn(self.hidden1_size, self.hidden2_size)
        self.weights3 = np.random.randn(self.hidden2_size, self.output_size)

        # Инициализация смещений (bias)
        self.bias1 = np.zeros((1, self.hidden1_size))
        self.bias2 = np.zeros((1, self.hidden2_size))
        self.bias3 = np.zeros((1, self.output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Прямое распространение
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        output = self.softmax(self.z3)
        return output

    def backward(self, X, y, output):
        # Обратное распространение ошибки
        output_error = output - y  # Ошибка на выходе
        output_delta = output_error  # Нет активации softmax

        # Ошибка скрытых слоев
        z2_error = np.dot(output_delta, self.weights3.T)
        z2_delta = z2_error * (self.z2 > 0)  # Производная ReLU

        z1_error = np.dot(z2_delta, self.weights2.T)
        z1_delta = z1_error * (self.z1 > 0)  # Производная ReLU

        # Обновление весов и смещений
        self.weights3 -= self.lr * np.dot(self.a2.T, output_delta)
        self.bias3 -= self.lr * np.sum(output_delta, axis=0, keepdims=True)

        self.weights2 -= self.lr * np.dot(self.a1.T, z2_delta)
        self.bias2 -= self.lr * np.sum(z2_delta, axis=0, keepdims=True)

        self.weights1 -= self.lr * np.dot(X.T, z1_delta)
        self.bias1 -= self.lr * np.sum(z1_delta, axis=0, keepdims=True)

    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if (epoch + 1) % 25 == 0:
                loss = -np.mean(y * np.log(output + 1e-8))
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    # Функция для сохранения весов
    def save_weights(self, file_path):
        np.savez(file_path,
                 weights1=self.weights1,
                 weights2=self.weights2,
                 weights3=self.weights3,
                 bias1=self.bias1,
                 bias2=self.bias2,
                 bias3=self.bias3)
        print(f"Weights saved to {file_path}")

    # Функция для загрузки весов
    def load_weights(self, file_path):
        data = np.load(file_path)
        self.weights1 = data['weights1']
        self.weights2 = data['weights2']
        self.weights3 = data['weights3']
        self.bias1 = data['bias1']
        self.bias2 = data['bias2']
        self.bias3 = data['bias3']
        print(f"Weights loaded from {file_path}")