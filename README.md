# Quantum Recommender System

A quantum similarity search implementation using QRAM, Swap Test, and Grover's algorithm for quantum recommender systems.

## Project Structure and Implementation Details

### File Structure

```
vibe/
├── qram/
│   ├── __init__.py
│   └── qram.py         # QRAM implementation
├── swap_test/
│   ├── __init__.py
│   └── swap_test.py    # Swap Test implementation
├── grover/
│   ├── __init__.py
│   └── grover.py       # Grover's algorithm implementation
├── main.py             # Command-line interface
└── various test files  # Unit and integration tests
```

### QRAM Module (qram/qram.py)

The QRAM module implements quantum random access memory for storing and retrieving quantum data.

**Class: QRAM**
- **Purpose**: Maps indices to data vectors using quantum operations
- **Key Methods**:
  - `__init__(num_index_qubits, num_data_qubits)`: Initializes the QRAM with specified dimensions
  - `load_database(database)`: Loads classical vectors into quantum superposition
  - `_amplitude_encoding(vector, wires)`: Encodes classical vectors into qubit amplitudes
  - `_selective_state_preparation(target_idx, vector)`: Prepares specific states conditional on index
  - `build_lookup_circuit()`: Creates quantum circuit that performs data lookup based on index

**Technical Implementation**:
- Uses controlled operations to selectively load data based on index register state
- Maps binary index strings to quantum amplitude-encoded data vectors
- Ensures data normalization for proper quantum amplitude encoding
- Operates on separate index and data wire registries

### Swap Test Module (swap_test/swap_test.py)

The Swap Test module implements quantum similarity measurement between quantum states.

**Class: SwapTest**
- **Purpose**: Quantifies similarity between two quantum states
- **Key Methods**:
  - `build_circuit(state1_wires, state2_wires, ancilla_wire)`: Creates swap test circuit

**Technical Implementation**:
- Applies Hadamard gate to ancilla qubit
- Performs controlled-SWAP operations between corresponding qubits in the two states
- Final Hadamard on ancilla creates interference pattern reflecting state similarity
- Probability of measuring ancilla as |0⟩ relates to state overlap: p(|0⟩) = (1 + |⟨ψ|φ⟩|²)/2
- Identical states yield p(|0⟩) = 1, orthogonal states yield p(|0⟩) = 0.5

### Grover Search Module (grover/grover.py)

The Grover Search module implements quantum search with amplitude amplification.

**Class: GroverSearch**
- **Purpose**: Finds database entries similar to a target vector using quantum amplitude amplification
- **Key Methods**:
  - `_initialize_omega(omega)`: Initializes target state
  - `_initialize_index_register()`: Creates superposition of all possible indices
  - `_oracle()`: Marks states similar to target omega using QRAM and swap test
  - `_grover_iteration()`: Performs one iteration of Grover's algorithm
  - `build_circuit(database, omega, iterations)`: Constructs complete quantum circuit
  - `search(database, omega, num_shots, iterations)`: Executes search and returns results

**Technical Implementation**:
- Integrates QRAM and SwapTest in the oracle function
- Oracle implements phase-flip on states where swap test indicates high similarity
- Diffusion operator implements reflection about average amplitude
- Wire management ensures proper connectivity between index, data, omega, and ancilla qubits
- Optimal number of iterations calculated as O(√N) where N is database size

### Main Program (main.py)

Command-line interface that coordinates the quantum recommender components.

**Implementation Details**:
- Parses command-line arguments for target vector, iterations, and shots
- Initializes the Grover search with appropriate dimensions
- Executes quantum search and collects measurement results
- Displays quantum and classical similarity results for comparison
- Database contains 4 vectors indexed by 2-qubit states (00, 01, 10, 11)

## Working Principle

Full working principle can be found at: [WORKING_PRINCIPLE](WORKING_PRINCIPLE.md)
