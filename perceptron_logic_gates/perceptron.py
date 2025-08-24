import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weight = np.zeros(input_size)   # only 2 inputs, no +1
        self.bias = 0.0
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        z = np.dot(self.weight, inputs) + self.bias
        return self.activation(z)

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                # Update rule (vectorized)
                self.weight += self.lr * error * inputs
                self.bias += self.lr * error
                total_error += abs(error)
            print(f"Epoch {epoch+1}/{epochs}, Total Error: {total_error}")
        print(f"Final Weights: {self.weight}, Bias: {self.bias}")
        print("Training completed!\n")