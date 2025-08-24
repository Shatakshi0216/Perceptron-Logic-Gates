🧠 Perceptron Logic Gate Challenge
Live Demo: https://perceptron-logic-gates.streamlit.app

📜 Overview
This project is a solution to the "Build a Single Neuron" challenge from the DC Hackathon. It features a Perceptron neuron built entirely from scratch using Python and NumPy. The interactive web application, built with Streamlit, allows users to train the neuron on various logic gates (AND, OR, NAND, XOR), visualize the learning process in real-time, and test the trained model with custom inputs.

The demo serves as an educational tool to explore the fundamentals of neural networks, the concept of linear separability, and the limitations of a single neuron.

✨ Features
This application includes several unique features designed to provide an intuitive and insightful user experience:

Perceptron Built From Scratch: The core Perceptron class is implemented using only NumPy, demonstrating a fundamental understanding of the neuron's architecture and learning mechanism.

Interactive Training: Users can configure the training process by selecting the logic gate, number of epochs, and the learning rate, allowing them to experiment with different hyperparameters.

Live Animated Learning: Instead of just showing a final result, the application animates the decision boundary's adjustments across each epoch. This provides a clear visual representation of how the neuron "learns" to classify data.

Analysis of Linear Separability: The app automatically analyzes the training outcome, explaining why the neuron succeeds on linearly separable problems (AND, OR, NAND) and why it fails on non-linearly separable problems (XOR).

The "XOR Solver" Explanation: For the XOR gate, the app provides a clear, theoretical explanation of how a Multi-Layer Perceptron (MLP) overcomes the limitations of a single neuron by combining multiple decision boundaries to solve complex problems.

Live Prediction: After a model is successfully trained, a new UI section appears, allowing users to input their own values (0 or 1) and get an instant prediction from the neuron they just trained.

🔧 Technical Deep Dive
Why the Perceptron Learning Rule?
For this challenge, we used the classic Perceptron Learning Rule. This was a deliberate choice for several reasons:

Historical Significance: It's the original learning algorithm proposed by Frank Rosenblatt in 1957 and is the simplest, most direct way to train a single neuron.

Simplicity and Clarity: The rule is incredibly intuitive: if the prediction is wrong, nudge the weights and bias slightly in the direction that would have made the prediction correct. This is easy to implement and understand from first principles.

Guaranteed Convergence: For linearly separable data, the Perceptron Learning Rule is mathematically proven to converge to a correct solution in a finite number of steps.

Why Not Gradient Descent and a Cost Function?
While Gradient Descent is the backbone of modern deep learning, it is not suitable for a simple Perceptron that uses a step activation function (like ours does). Here's why:

A step function is not continuously differentiable. Its derivative is zero everywhere except at the point of the step, where it is undefined.

Gradient Descent relies on calculating the gradient (derivative) of a cost function with respect to the weights. Since our activation function's derivative is almost always zero, the gradient would also be zero.

A zero gradient means the model receives no information about which direction to update the weights, and therefore, it cannot learn.

The Perceptron Learning Rule cleverly sidesteps this problem by not relying on gradients. It updates weights based on the raw error (target - prediction), providing a simple yet effective learning mechanism for this specific type of neuron.

🚀 How to Run Locally
To run this application on your local machine, please follow these steps:

1. Prerequisites:

Python 3.7 or higher

Git

2. Clone the Repository:

git clone https://github.com/your-username/Perceptron-Logic-Gates.git
cd Perceptron-Logic-Gates

3. Create a Virtual Environment (Recommended):

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies:
The requirements.txt file contains all the necessary packages.

pip install -r requirements.txt

5. Run the Streamlit App:

streamlit run app.py

The application should now be open and running in your web browser!

📁 File Structure
The repository is organized to separate the core model logic from the application interface, which is a standard best practice.

Perceptron-Logic-Gates/
├── .gitignore          # Ignores unnecessary files
├── README.md           # This file
├── requirements.txt    # Project dependencies (streamlit, numpy, matplotlib)
├── app.py              # The main file for the Streamlit application
└── perceptron_logic_gates/
    ├── __init__.py     # Makes this directory a Python package
    ├── perceptron.py   # Contains the Perceptron class (the model)
    └── gates.py        # Defines the logic gate datasets
