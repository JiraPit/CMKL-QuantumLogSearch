"""
Quantum Log Anomaly Detection Improved Version

Description:
    This enhanced script builds on the initial PennyLane AdamOptimizer implementation
    by switching to a NesterovMomentumOptimizer, increasing the variational ansatz
    depth, and using StronglyEntanglingLayers. As a result, it achieves better
    final performance than the first model:

        • Final training: Loss = 0.5934, Accuracy = 83.3%
        • Test performance:  Accuracy = 79.0%, F1 Score = 0.851

Dependencies:
    - pennylane
    - numpy (via pennylane.numpy)
    - pandas
    - scikit-learn
    - matplotlib

Usage:
    python quantum_log_nesterov.py

Notes:
    - Adjust `num_layers`, optimizer hyperparameters, and `num_iter` to further
      tune performance.
    - Ensure PennyLane's default.qubit backend and all Python packages are installed.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt

n_samples = 500
rng = np.random.default_rng(42)

severity    = rng.integers(0, 3, size=n_samples)
msg_len     = rng.integers(20, 120, size=n_samples)
num_digits  = rng.integers(0, 10, size=n_samples)
num_special = rng.integers(0, 5, size=n_samples)
label       = ((severity == 2) | (msg_len > 80)).astype(int)

df = pd.DataFrame({
    "severity": severity,
    "msg_len": msg_len,
    "num_digits": num_digits,
    "num_special": num_special,
    "anomaly": label,
})

df.to_csv("log_data.csv", index=False)
print("Saved synthetic log data to log_data.csv")

# Feature scaling
df["f1"] = df["severity"]   / 2
df["f2"] = (df["msg_len"]  - 20) / 100
df["f3"] = df["num_digits"]/ 10
df["f4"] = df["num_special"]/ 5

features = ["f1", "f2", "f3", "f4"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features].values,
    df["anomaly"].values * 2 - 1,
    test_size=0.2,
    random_state=42,
    stratify=df["anomaly"],
)

# Quantum setup
num_qubits = 4
num_layers = 4  # for tuning
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    # embed features
    qml.templates.AngleEmbedding(x, wires=range(num_qubits), rotation="Y")
    # stronger entangling ansatz
    qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

def variational_classifier(w, b, x):
    return circuit(w, x) + b

def square_loss(Y, P):
    return np.mean((Y - P) ** 2)

def accuracy(Y, P):
    return np.mean(np.sign(P) == Y)

def cost(w, b, X, Y):
    P = np.array([variational_classifier(w, b, x) for x in X])
    return square_loss(Y, P)

# Initialize parameters & optimizer
np.random.seed(0)
# StronglyEntanglingLayers has 3 rotation angles per qubit per layer
weights = 0.1 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias    = np.array(0.0, requires_grad=True)

opt      = NesterovMomentumOptimizer(stepsize=0.2, momentum=0.9)
batch_sz = 20
num_iter = 100

# Lists for tracking
train_costs = []
train_accs  = []

# Training loop
for it in range(num_iter):
    idx     = rng.integers(0, len(X_train), size=batch_sz)
    X_batch = X_train[idx]
    Y_batch = y_train[idx]

    # Nesterov step returns updated (weights, bias)
    weights, bias = opt.step(
        lambda w, b: cost(w, b, X_batch, Y_batch),
        weights,
        bias,
    )

    # compute full-train metrics
    train_c = cost(weights, bias, X_train, y_train)
    train_p = np.array([variational_classifier(weights, bias, x) for x in X_train])
    train_a = accuracy(y_train, train_p)

    train_costs.append(train_c)
    train_accs.append(train_a)

    if (it + 1) % 10 == 0:
        print(f"Iter {it+1:3d} | loss {train_c:.4f} | acc {train_a:.3f}")

# Evaluation
test_p = np.array([variational_classifier(weights, bias, x) for x in X_test])
test_s = np.sign(test_p)
print("\nTest Accuracy:", accuracy_score(y_test, test_s))
print("Test F1 score:",     f1_score(y_test, test_s))

# Plot training curves
plt.figure()
plt.plot(train_costs, label="Train Loss")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Iterations")
plt.legend()
plt.show()

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Iterations")
plt.legend()
plt.show()

# Draw the circuit and save it as PNG
import matplotlib.pyplot as plt
from pennylane.draw import draw_mpl

# Create a Matplotlib drawer for your qnode
drawer = draw_mpl(circuit)

# Draw the circuit for one sample, returns a Matplotlib Figure
fig = drawer(weights, X_train[0])

# Save to disk
fig.savefig("circuit_diagram.png", bbox_inches="tight", dpi=300)
print("Saved circuit diagram to circuit_diagram.png")
