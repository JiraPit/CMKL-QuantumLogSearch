# Quantum Recommendation System: Working Principles

This document explains the working principles of our quantum recommendation system, particularly focusing on the QRAM (Quantum Random Access Memory) integration with Grover's search algorithm for similarity search.

## 1. System Overview

Our quantum recommendation system uses three primary components:
- **QRAM**: Stores and retrieves database entries in quantum superposition
- **Grover's Algorithm**: Amplifies the probability of finding database entries similar to a target vector
- **Swap Test**: Measures similarity between quantum states

## 2. QRAM Operation

### 2.1 Database Loading

The QRAM stores a classical database as a dictionary mapping binary indices to normalized vectors:

```python
database = {
    "00": [1.0, 0.0, 0.0, 0.0],  # |00⟩
    "01": [0.0, 1.0, 0.0, 0.0],  # |01⟩
    "10": [0.0, 0.0, 1.0, 0.0],  # |10⟩
    "11": [0.0, 0.0, 0.0, 1.0],  # |11⟩
}
```

The `load_database()` method:
- Converts integer indices to binary strings
- Verifies that all vectors have the same dimensions
- Normalizes vectors for amplitude encoding
- Stores them in the QRAM's internal database

### 2.2 Data Retrieval Process

The key quantum advantage comes from superposition:

1. **Index Register Preparation**:
   - The index register is initialized to equal superposition of all possible indices:
   - |ψ_index⟩ = 1/√N ∑|i⟩
   - For example, with 2 qubits: |ψ_index⟩ = 1/2(|00⟩ + |01⟩ + |10⟩ + |11⟩)

2. **Selective State Preparation**:
   - For each index-vector pair in the database, apply a controlled operation
   - This creates an entangled state between index and data registers:
   - |ψ_entangled⟩ = 1/√N ∑|i⟩|vector_i⟩
   - Each data vector is loaded into the data register, but only for the component of the superposition where the index matches

3. **Parallel Data Access**:
   - Thanks to quantum superposition, all database entries are loaded simultaneously
   - The data is accessed in parallel rather than sequentially

## 3. Grover's Search Algorithm

### 3.1 Oracle Function

The oracle marks database entries similar to the target vector (omega):

1. **QRAM Lookup**:
   - The QRAM loads all vectors in superposition: 1/√N ∑|i⟩|vector_i⟩

2. **Swap Test**:
   - Compares the data register with the omega register
   - Creates the state: 1/√N ∑|i⟩|vector_i⟩|similarity_i⟩
   - The ancilla qubit encodes similarity information:
     - When states are identical: ancilla → |0⟩ with probability 1
     - When states are orthogonal: ancilla → |0⟩ or |1⟩ with probability 0.5
     - For partially similar states: probability of |0⟩ is p(|0⟩) = (1 + |⟨vector_i|ω⟩|²)/2

3. **Marking Similar States**:
   - Apply X gate to flip the ancilla: |0⟩ ↔ |1⟩
   - Now indices with vectors similar to omega have ancilla in state |1⟩
   - Apply controlled-Z operations to index register bits when ancilla is |1⟩
   - This applies a phase flip to indices with vectors similar to omega

4. **Uncomputation**:
   - The ancilla is uncomputed to disentangle it from the rest of the system

### 3.2 Grover Iteration

Each Grover iteration consists of:
1. Oracle application: Marks states similar to the target
2. Diffusion operator: Amplifies the marked states

After multiple iterations, measuring the index register gives a high probability of finding indices where the corresponding vector is similar to omega.

## 4. Complete Workflow

1. **Initialization**:
   - Load database into QRAM
   - Initialize omega register with target vector
   - Initialize index register to equal superposition

2. **Grover Iterations**:
   - For each iteration:
     - Apply oracle to mark similar vectors
     - Apply diffusion operator to amplify marked states

3. **Measurement**:
   - Measure index register
   - The most frequently observed index corresponds to the database entry most similar to omega
