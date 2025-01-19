import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pennylane as qml

# Load EEG data (example dataset URL)
data_path = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf"
try:
    import mne
    raw = mne.io.read_raw_edf(data_path, preload=True)
    raw.filter(7., 30.)  # Bandpass filter (7-30 Hz)
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.5)
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    labels = np.random.choice([0, 1], size=(data.shape[0],))  # Example labels
except:
    print("Dataset access failed. Please download manually or adjust dataset source.")
    data = np.random.rand(100, 8, 100)  # Simulated data (100 samples, 8 channels, 100 time points)
    labels = np.random.choice([0, 1], size=(100,))

# Preprocessing
data = data[:, :4, :50]  # Select first 4 channels and reduce time points
data = data.reshape(data.shape[0], -1)  # Flatten each sample
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define quantum device
n_qubits = 4  # Match the number of selected EEG channels
dev = qml.device("default.qubit", wires=n_qubits)

# Improved quantum circuit with additional entangling layers
def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)  # Apply RY rotation for feature encoding

@qml.qnode(dev)
def quantum_circuit(weights, x):
    feature_map(x)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))  # Add more entangling layers
    return qml.expval(qml.PauliZ(0))  # Output from the first qubit

# Initialize weights with a smaller range for better gradient flow
n_layers = 4  # Increase the number of layers for more expressiveness
weights = np.random.random((n_layers, n_qubits, 3), requires_grad=True)

# Define cost function (using mean squared error)
def cost_function(weights, X, y):
    predictions = [quantum_circuit(weights, x) for x in X]
    return np.mean((np.array(predictions) - y) ** 2)

# Training with better optimization
opt = qml.AdamOptimizer(stepsize=0.05)  # Slightly lower learning rate for more stable training
steps = 100  # Increase number of steps for better convergence

# Training loop
for step in range(steps):
    weights, cost = opt.step_and_cost(lambda w: cost_function(w, X_train, y_train), weights)
    if step % 10 == 0:
        print(f"Step {step}: Cost = {cost:.4f}")

# Evaluate
predictions = [np.sign(quantum_circuit(weights, x)) for x in X_test]
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.2f}")
