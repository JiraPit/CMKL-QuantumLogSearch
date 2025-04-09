# Quantum Recommender System

A quantum algorithm implementation that uses quantum computing for similarity search and recommendations.

## Overview

This system leverages quantum computing concepts to implement a similarity search algorithm using Grover's search. The system uses a quantum oracle that checks if the cosine similarity between a state and a target exceeds a specified threshold.

## Architecture

The system is organized into modular components:

- **Main** (`main.py`): Contains the `QuantumRecommender` class that orchestrates the system
- **Embedding**:
  - `sentence_embedding.py`: Handles embedding of sentences into vectors
  - `q_state_embedding.py`: Handles embedding classical data into quantum amplitudes
- **Similarity Search**:
  - `circuit_builder.py`: Builds quantum circuits for similarity search
  - `simulator.py`: Manages Grover algorithm simulations
- **State**:
  - `quantum_state.py`: Creates and manipulates quantum states

## Dependencies

- Qiskit
- Qiskit-Aer
- NumPy
- Sentence Transformers
- Math

## Usage

The system has been designed with a dependency injection pattern for configuration. The `QuantumRecommender` class is the main entry point:

```python
from main import QuantumRecommender, DEFAULT_CONFIG

# Use with default configuration
recommender = QuantumRecommender()

# Or with custom configuration
custom_config = DEFAULT_CONFIG.copy()
custom_config["NUM_QUBITS"] = 3
custom_config["THRESHOLD"] = 0.2
recommender = QuantumRecommender(config=custom_config)

# Run similarity search
results = recommender.run_high_similarity_search()

# Use sentence embedding
sentences = ["Quantum computing uses qubits", "Machine learning is powerful"]
embeddings = recommender.embed_sentences(sentences)
```

See `example.py` for more detailed usage examples.

## Configuration Parameters

- `NUM_QUBITS`: Number of qubits to use in quantum circuits (default: 5)
- `THRESHOLD`: Similarity threshold for search (default: 0.1)
- `SHOTS`: Number of simulation shots (default: 1024)
- `EMBEDDING_MODEL`: Default sentence embedding model (default: "all-MiniLM-L6-v2")

