# Quantum Recommender System

A quantum algorithm implementation that uses quantum computing for similarity search and recommendations based on article content.

## Overview

This system leverages quantum computing concepts to implement a similarity search algorithm for recommending articles. It embeds article text into quantum states and uses quantum state fidelity (overlap) to find articles with similar content.

## Architecture

The system is organized into modular components:

- **Main** (`main.py`): Contains the `QuantumRecommender` class that orchestrates the system and provides the main interface
- **Embedding**:
  - `sentence_embedding.py`: Handles embedding of text content into vectors using Sentence Transformers
  - `q_state_embedding.py`: Embeds classical vectors into quantum amplitudes
- **Similarity Search**:
  - `circuit_builder.py`: Builds quantum circuits for similarity search
  - `simulator.py`: Manages Grover algorithm simulations
  - `utils/create_state.py`: Utility functions for quantum state creation

## Dataset

The system uses an article database CSV file that contains:
- Index
- Article full name
- Article description
- Article body

## Dependencies

- Qiskit and Qiskit-Aer for quantum simulation
- NumPy for numerical operations
- Pandas for data manipulation
- Sentence Transformers for text embedding

## Usage

To run the recommendation system:

```python
# Run the main script
python main.py
```

The system will:
1. Load the article database
2. Generate embeddings for all articles
3. Prompt the user to select an article by index
4. Display the selected article details
5. Show recommended similar articles based on quantum similarity
6. Allow the user to select from recommended articles to continue exploration

## How It Works

1. **Text Embedding**: Article text (title + description) is converted into semantic vectors
2. **Quantum Embedding**: Vectors are embedded into quantum states
3. **Similarity Calculation**: Quantum fidelity between states is used as similarity measure
4. **Recommendation**: Articles with highest similarity to the selected article are recommended