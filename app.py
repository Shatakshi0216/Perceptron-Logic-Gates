import streamlit as st
import numpy as np
import pandas as pd
from perceptron_logic_gates.perceptron import Perceptron
from perceptron_logic_gates.gates import X, LOGIC_GATES

# --- App Layout ---
st.set_page_config(layout="wide")
st.title("🧠 Perceptron Logic Gate Challenge")
st.write(
    "An implementation of a single neuron from scratch to learn basic logic gates, "
    "as per Problem Statement 12 of the DC Hackathon."
)

# --- Sidebar for User Input ---
st.sidebar.header("Configuration")
selected_gate = st.sidebar.selectbox("Choose a Logic Gate to Train:", list(LOGIC_GATES.keys()))
epochs = st.sidebar.slider("Select Number of Training Epochs:", 1, 20, 10)
lr = st.sidebar.number_input("Learning Rate:", 0.01, 1.0, 0.1, 0.01)

# --- Training and Display Logic ---
if st.sidebar.button("🚀 Train Neuron"):
    st.subheader(f"Training for: {selected_gate} Gate")

    # Get data for the selected gate
    y = LOGIC_GATES[selected_gate]

    # Initialize and train the Perceptron
    p = Perceptron(input_size=2, lr=lr)

    # Display training progress
    st.write("Training in progress...")
    with st.expander("Show Training Log"):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = p.predict(inputs)
                error = target - prediction
                p.weight += p.lr * error * inputs
                p.bias += p.lr * error
                total_error += abs(error)
            st.text(f"Epoch {epoch+1}/{epochs}, Total Error: {total_error}")
            if total_error == 0:
                st.success("Convergence reached! The data is linearly separable.")
                break

    st.success("Training complete!")
    st.write(f"**Final Weights:** `{p.weight}` | **Final Bias:** `{p.bias:.2f}`")

    # --- Display Results ---
    st.subheader("Results")

    # Get final predictions
    predictions = [p.predict(i) for i in X]

    # Create a results table
    results_df = pd.DataFrame({
        'Input 1': X[:, 0],
        'Input 2': X[:, 1],
        'Target': y,
        'Prediction': predictions
    })
    results_df['Correct ✅'] = results_df['Target'] == results_df['Prediction']
    st.dataframe(results_df)

    # --- Analysis ---
    st.subheader("Analysis")
    if selected_gate == "XOR":
        st.error(
            "**The Perceptron failed to learn the XOR gate.** This is expected! "
            "The XOR data is not linearly separable, meaning a single straight line cannot "
            "separate the `0`s from the `1`s. This demonstrates a key limitation of a single neuron."
        )
    else:
        st.success(
            f"**The Perceptron successfully learned the {selected_gate} gate.** "
            f"This is because the {selected_gate} data is linearly separable, and the Perceptron "
            "was able to find a decision boundary to correctly classify all inputs."
        )

else:
    st.info("Select a logic gate and click 'Train Neuron' to begin.")