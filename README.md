# 🧠 Perceptron Logic Gate Challenge  

🔗 **Live Demo:** [Perceptron Logic Gates App](https://perceptron-logic-gates.streamlit.app)  

---

## 📜 Overview  
This project is a solution to the **"Build a Single Neuron" challenge** from the DC Hackathon.  
It features a **Perceptron neuron built entirely from scratch** using Python and NumPy.  

The interactive web application, built with **Streamlit**, allows users to:  
- Train the neuron on various logic gates (**AND, OR, NAND, XOR**)  
- Visualize the learning process in real-time  
- Test the trained model with custom inputs  

This demo serves as an **educational tool** to explore:  
- Fundamentals of neural networks  
- Concept of **linear separability**  
- **Limitations** of a single neuron  

---

## ✨ Features  

✅ **Perceptron Built From Scratch**  
- Implemented using only NumPy  
- Demonstrates the neuron's architecture and learning mechanism  

✅ **Interactive Training**  
- Configure logic gate, epochs, and learning rate  
- Experiment with hyperparameters to observe learning behavior  

✅ **Live Animated Learning**  
- Visualizes decision boundary updates across epochs  
- Clearly shows how the neuron "learns"  

✅ **Linear Separability Analysis**  
- Explains success on linearly separable gates (AND, OR, NAND)  
- Explains failure on non-linearly separable gates (XOR)  

✅ **XOR Solver Explanation**  
- Theoretical explanation of how an **MLP (Multi-Layer Perceptron)** overcomes XOR limitations  

✅ **Live Predictions**  
- After training, users can input **custom 0/1 values**  
- Instant predictions from the trained neuron  

---

## 🔧 Technical Deep Dive  

### Why the Perceptron Learning Rule?  
- **Historical Significance** → Proposed by Frank Rosenblatt in 1957, the first neural learning algorithm.  
- **Simplicity** → Easy to implement and understand: update weights when predictions are wrong.  
- **Convergence Guarantee** → For linearly separable data, guaranteed to converge in finite steps.  

### Why Not Gradient Descent?  
- Uses a **step activation function**, which is **non-differentiable**.  
- Derivative = 0 almost everywhere → gradient vanishes.  
- Perceptron Learning Rule sidesteps this problem → updates based on **raw error (target - prediction)**.  

---

## 🚀 How to Run Locally  

### 1. Prerequisites  
- Python **3.7+**  
- Git  

### 2. Clone Repository  
```bash
git clone https://github.com/shivamr021/Perceptron-Logic-Gates.git
cd Perceptron-Logic-Gates
````

### 3. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

The app should open in your browser automatically 🎉

---

## 📁 File Structure

```
Perceptron-Logic-Gates/
├── .gitignore             # Ignore unnecessary files
├── README.md              # Project documentation
├── requirements.txt       # Dependencies (streamlit, numpy, matplotlib)
├── app.py                 # Main Streamlit application
└── perceptron_logic_gates/
    ├── __init__.py        # Makes this a Python package
    ├── perceptron.py      # Perceptron class (model logic)
    └── gates.py           # Logic gate datasets
```

---

## 📊 Example Visuals

Here’s what you’ll see in the app:

* Decision boundary evolving per epoch
* Final trained classification regions
* Explanations for success/failure depending on gate type

---

## 📚 Learning Outcomes

Through this project, you will:

* Understand **how a Perceptron works**
* See **why XOR cannot be solved** by a single neuron
* Gain intuition about **linear separability**
* Experiment with **training dynamics** interactively

---

## 🛠️ Tech Stack

* **Python** (Core implementation)
* **NumPy** (Matrix operations & model logic)
* **Matplotlib** (Decision boundary visualization)
* **Streamlit** (Interactive web app)

---

## 🚩 Future Improvements

* ✅ Add **Multi-Layer Perceptron (MLP)** implementation for XOR
* ✅ Include **downloadable training logs**
* ✅ Add support for **custom datasets** beyond logic gates
* ✅ Provide **step-by-step math explanations** alongside training visuals

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. Fork the repo
2. Create a new branch (`feature-xyz`)
3. Commit your changes
4. Push the branch & create a PR

---

## 🧑‍💻 Author

- **Shivam Rathod**  
  🔗 [GitHub](https://github.com/shivamr021) • [LinkedIn](https://linkedin.com/in/shatakshitiwari017)  

- **Shatakshi Tiwari**  
  🔗 [GitHub](https://github.com/Shatakshi0216) • [LinkedIn](https://linkedin.com/in/shivamrathod021)  

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it with attribution.

---

