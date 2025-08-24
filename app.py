import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from perceptron_logic_gates.perceptron import Perceptron
from perceptron_logic_gates.gates import X, LOGIC_GATES

# --- App Layout ---
st.set_page_config(layout="wide")
st.title("🧠 Perceptron Logic Gate Challenge")
st.write(
    "An implementation of a single neuron from scratch to learn basic logic gates. "
    "This demo animates the learning process to show how the decision boundary is found."
)

# --- Sidebar for User Input ---
st.sidebar.header("Configuration")
selected_gate = st.sidebar.selectbox("Choose a Logic Gate to Train:", list(LOGIC_GATES.keys()))
epochs = st.sidebar.slider("Select Number of Training Epochs:", 1, 30, 10)
lr = st.sidebar.number_input("Learning Rate:", 0.01, 1.0, 0.1, 0.01)

# --- Main App Logic ---
if st.sidebar.button("🚀 Train Neuron"):
    
    # Create columns for the layout
    col1, col2 = st.columns([1, 1]) # Give equal width to columns

    with col1:
        st.subheader(f"Training Animation: {selected_gate} Gate")
        # Create a placeholder for our animation plot inside the column
        plot_placeholder = st.empty()

    # Get data and initialize the Perceptron
    y = LOGIC_GATES[selected_gate]
    p = Perceptron(input_size=2, lr=lr)

    # Main training loop for animation
    for epoch in range(epochs):
        total_error = 0
        # Update weights and bias based on each data point
        for inputs, target in zip(X, y):
            prediction = p.predict(inputs)
            error = target - prediction
            p.weight += p.lr * error * inputs
            p.bias += p.lr * error
            total_error += abs(error)

        # --- Animation Frame Generation ---
        # Create a figure with a controlled size
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Plot data points (Blue for 0, Red for 1)
        colors = ['blue' if target == 0 else 'red' for target in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.9, edgecolors='k')
        
        # Plot the CURRENT decision boundary
        x_vals = np.array([-0.5, 1.5]) # Use fixed x-range for consistency
        if p.weight[1] != 0:
            y_vals = (-p.weight[0] * x_vals - p.bias) / p.weight[1]
            ax.plot(x_vals, y_vals, '--', color='gray', label='Decision Boundary')
        elif p.weight[0] != 0: # Handle vertical line
            ax.axvline(x=-p.bias / p.weight[0], linestyle='--', color='gray', label='Decision Boundary')

        # Style the plot
        ax.set_xlim([-0.5, 1.5]); ax.set_ylim([-0.5, 1.5])
        ax.set_title(f"Epoch: {epoch + 1}/{epochs} | Total Error: {total_error}")
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        
        # Update the placeholder with the new plot
        with col1:
             plot_placeholder.pyplot(fig)
        plt.close(fig) # Prevent plots from stacking up in memory
        
        # Check for convergence
        if total_error == 0:
            st.success(f"Convergence reached at Epoch {epoch + 1}!")
            break
        
        time.sleep(0.25) # Control animation speed

    # --- Display Final Results and Analysis AFTER the loop ---
    with col2:
        st.subheader("Analysis")
        if selected_gate == "XOR" and total_error > 0:
            st.error(
                "**The Perceptron failed to learn the XOR gate.** As the animation shows, "
                "a single straight line cannot separate the red and blue points. "
                "This proves that the XOR problem is not linearly separable."
            )
        elif total_error == 0:
            st.success(
                f"**The Perceptron successfully learned the {selected_gate} gate.** "
                "The animation shows the decision boundary adjusting until it "
                "perfectly separated the two classes of data points."
            )
        else:
            st.warning(
                "**Training finished, but the model did not fully converge.** "
                "Try increasing the number of epochs or adjusting the learning rate."
            )
        
        # Use markdown with inline CSS to reduce the top margin
        st.markdown("<h3 style='margin-top: -15px;'>Final Results</h3>", unsafe_allow_html=True)
        predictions = [p.predict(i) for i in X]
        # Create a markdown table (no pandas)
        table = "| Input 1 | Input 2 | Target | Prediction |\n|---|---|---|---|\n"
        for i in range(len(X)):
            table += f"| {X[i][0]} | {X[i][1]} | {y[i]} | {predictions[i]} |\n"
        st.markdown(table)
        st.write(f"**Final Weights:** `{np.round(p.weight, 2)}` | **Final Bias:** `{p.bias:.2f}`")

else:
    st.info("Select a logic gate and click 'Train Neuron' to begin the animated training.")

