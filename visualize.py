import matplotlib.pyplot as plt

def plot_decision_boundary(perceptron, X, y, title="Decision Boundary"):
    # scatter plot of inputs
    for inp, target in zip(X, y):
        color = "blue" if target == 0 else "red"
        plt.scatter(inp[0], inp[1], color=color)

    # decision boundary: w1*x1 + w2*x2 + b = 0
    x1 = [0, 1]
    x2 = [(-perceptron.weights[0]*xi - perceptron.bias) / perceptron.weights[1] for xi in x1]
    plt.plot(x1, x2, 'k-')

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
