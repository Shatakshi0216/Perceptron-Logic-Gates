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
    "This demo animates the learning process and shows how to solve non-linear problems."
)

# --- Initialize Session State ---
if 'last_run_was_failed_xor' not in st.session_state:
    st.session_state.last_run_was_failed_xor = False
if 'show_xor_solution' not in st.session_state:
    st.session_state.show_xor_solution = False

def reset_state():
    """Callback function to reset all relevant states."""
    st.session_state.last_run_was_failed_xor = False
    st.session_state.show_xor_solution = False

# --- Sidebar for User Input ---
st.sidebar.header("Configuration")
selected_gate = st.sidebar.selectbox(
    "Choose a Logic Gate to Train:",
    list(LOGIC_GATES.keys()),
    on_change=reset_state
)
epochs = st.sidebar.slider("Select Number of Training Epochs:", 1, 30, 10)
lr = st.sidebar.number_input("Learning Rate:", 0.01, 1.0, 0.1, 0.01)

# --- Main App Logic ---

# Check the app's state first to decide on the layout
if st.session_state.show_xor_solution:
    # --- FULL-WIDTH XOR SOLUTION DISPLAY ---
    st.subheader("The Solution: A Multi-Layer Perceptron (MLP)")
    st.info(
        """
        **Yes, by combining neurons!** A single Perceptron can only create one straight line. 
        The XOR problem is unsolvable with one line, which is why our model failed.

        The solution is to use a **Multi-Layer Perceptron (MLP)**, which is the foundation of modern neural networks. Here’s how it works:

        1.  **Hidden Layer:** An MLP has one or more "hidden layers" of neurons between the input and output. Each neuron in this layer learns its own linear boundary (its own straight line).

        2.  **Combining Boundaries:** The final output neuron doesn't look at the raw inputs; instead, it looks at the outputs of the hidden neurons. It learns to combine their simple lines to form a complex, non-linear decision region.

        Essentially, an MLP can solve the XOR problem by learning that the answer is '1' if the inputs fall on one side of the first hidden neuron's line **AND** on the other side of the second hidden neuron's line. This creates the exact non-linear separation needed.
        """
    )

else:
    # --- TWO-COLUMN LAYOUT FOR TRAINING AND RESULTS ---
    col1, col2 = st.columns([1, 1])

    with col1:
        subheader_placeholder = st.empty()
        plot_placeholder = st.empty()

    with col2:
        analysis_placeholder = st.empty()
        results_placeholder = st.empty()
        xor_button_placeholder = st.empty()

    if st.sidebar.button("🚀 Train Neuron"):
        reset_state() # Reset state before a new training run

        subheader_placeholder.subheader(f"Training Animation: {selected_gate} Gate")
        y = LOGIC_GATES[selected_gate]
        p = Perceptron(input_size=2, lr=lr)

        # Main training loop
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = p.predict(inputs)
                error = target - prediction
                p.weight += p.lr * error * inputs
                p.bias += p.lr * error
                total_error += abs(error)

            fig, ax = plt.subplots(figsize=(7, 6))
            colors = ['blue' if target == 0 else 'red' for target in y]
            ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.9, edgecolors='k')

            x_vals = np.array([-0.5, 1.5])
            if p.weight[1] != 0:
                y_vals = (-p.weight[0] * x_vals - p.bias) / p.weight[1]
                ax.plot(x_vals, y_vals, '--', color='gray', label='Decision Boundary')
            elif p.weight[0] != 0:
                ax.axvline(x=-p.bias / p.weight[0], linestyle='--', color='gray', label='Decision Boundary')

            ax.set_xlim([-0.5, 1.5]); ax.set_ylim([-0.5, 1.5])
            ax.set_title(f"Epoch: {epoch + 1}/{epochs} | Total Error: {total_error}")
            ax.set_xlabel("Input 1"); ax.set_ylabel("Input 2")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.legend()

            plot_placeholder.pyplot(fig)
            plt.close(fig)

            if total_error == 0:
                st.success(f"Convergence reached at Epoch {epoch + 1}!")
                break

            time.sleep(0.25)

        # --- Populate Analysis and Results Placeholders ---
        with analysis_placeholder.container():
            st.subheader("Analysis")
            if selected_gate == "XOR" and total_error > 0:
                st.error(
                    "**The Perceptron failed to learn the XOR gate.** A single straight line cannot "
                    "separate the red and blue points, proving the problem is not linearly separable."
                )
                st.session_state.last_run_was_failed_xor = True
            elif total_error == 0:
                st.success(
                    f"**The Perceptron successfully learned the {selected_gate} gate.** "
                    "The decision boundary perfectly separates the two classes."
                )
            else:
                st.warning(
                    "**Training finished, but the model did not fully converge.** "
                    "Try increasing epochs or adjusting the learning rate."
                )

        with results_placeholder.container():
            st.markdown("<h3 style='margin-top: -15px;'>Final Results</h3>", unsafe_allow_html=True)
            predictions = np.array([p.predict(i) for i in X])
            accuracy = np.sum(predictions == y) / len(y) * 100
            st.metric(label="Model Accuracy", value=f"{accuracy:.0f}%")
            table = "| Input 1 | Input 2 | Target | Prediction |\n|---|---|---|---|\n"
            for i in range(len(X)):
                table += f"| {X[i][0]} | {X[i][1]} | {y[i]} | {predictions[i]} |\n"
            st.markdown(table)
            st.write(f"**Final Weights:** `{np.round(p.weight, 2)}` | **Final Bias:** `{p.bias:.2f}`")

    # --- XOR SOLVER BUTTON ---
    if st.session_state.last_run_was_failed_xor:
        if xor_button_placeholder.button("💡 Can we solve XOR?"):
            st.session_state.show_xor_solution = True
            st.session_state.last_run_was_failed_xor = False
            st.rerun()

    # Show initial message if no button has been clicked
    if not st.sidebar.is_touched:
        st.info("Select a logic gate and click 'Train Neuron' to begin the animated training.")
