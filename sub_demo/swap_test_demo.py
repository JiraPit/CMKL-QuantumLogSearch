"""
Swap Test Demo

This demo reads pre-embedded quantum states from a pickle file and performs
swap tests between randomly selected states to measure their similarity.
"""

# Import required modules
import pickle
import random
import numpy as np
from pathlib import Path
from qiskit import QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from embedding.q_state_embedding import QStateEmbedding
from similarity_search.circuit_builder import SimilaritySearchCircuitBuilder


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-10)


def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)


def main():
    """Main function to run the swap test demo."""
    print("Quantum Swap Test Demo")
    print("======================")

    # Check if pre-computed embeddings exist
    pre_computed_path = (
        Path(__file__).parent.parent / "pre_computed" / "pre_embedded_data.pkl"
    )

    if not pre_computed_path.exists():
        print("Pre-computed embeddings not found. Run pre_embed.py first.")
        return

    # Load pre-computed embeddings
    print("Loading pre-computed embeddings...")
    with open(pre_computed_path, "rb") as f:
        pre_computed_data = pickle.load(f)

    # Extract data from the pickle file
    embeddings = pre_computed_data["embeddings"]
    articles_df = pre_computed_data["filtered_df"]
    vector_size = pre_computed_data["vector_size"]

    # Determine number of qubits needed
    num_qubits = int(np.ceil(np.log2(vector_size)))
    print(f"Using {num_qubits} qubits for quantum state representation")

    # Get number of entries
    num_entries = len(articles_df)
    selected_indices = list(range(num_entries))

    # Create quantum states for the pre-computed embeddings
    print(f"Preparing quantum states for {num_entries} articles...")
    article_states = {}
    for i in selected_indices:
        state = QStateEmbedding.amplitude_encode(embeddings[i])
        article_states[i] = state

    print(f"Prepared states for {len(article_states)} articles")

    # Setup simulator
    shots = 1024

    # Perform swap tests
    num_tests = 10
    for i in range(num_tests):
        # Randomly select two article indices
        article_indices = list(article_states.keys())
        idx1, idx2 = random.sample(article_indices, 2)
        # idx1, idx2 = 217, 99

        # Get the quantum states
        state1 = article_states[idx1]
        state2 = article_states[idx2]

        # Print selected articles
        print(f"\nTest {i+1}/{num_tests}:")
        article1 = articles_df.iloc[idx1]
        article2 = articles_df.iloc[idx2]
        print(f"Article 1 [{idx1}]: {article1['doc_full_name']}")
        print(f"Article 2 [{idx2}]: {article2['doc_full_name']}")

        # Create registers for swap test
        ancilla_reg = QuantumRegister(1, "ancilla")
        state1_reg = QuantumRegister(num_qubits, "state1")
        state2_reg = QuantumRegister(num_qubits, "state2")
        cr = ClassicalRegister(1, "result")

        # Create circuit and initialize states
        qc = QuantumCircuit(ancilla_reg, state1_reg, state2_reg, cr)
        qc.initialize(state1, state1_reg)
        qc.initialize(state2, state2_reg)

        # Build swap test circuit
        swap_test = SimilaritySearchCircuitBuilder.swap_test_circuit(
            state1_reg,
            state2_reg,
            ancilla_reg,
        )

        # Build swap test circuit
        qc.compose(swap_test, inplace=True)

        # Measure ancilla qubit to classical bit
        qc.measure(ancilla_reg[0], cr[0])

        simulator = AerSimulator()
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate probability of measuring |0⟩
        p_zero = counts.get("0", 0) / shots

        # For SWAP test, similarity = 2*p_zero - 1
        similarity = 2 * p_zero - 1

        print("Measurement results:", counts)
        print(f"Similarity: {similarity:.4f}")
        print(f"Cosine similarity: {cosine_similarity(state1, state2):.4f}")
        print(f"Euclidean distance: {euclidean_distance(state1, state2):.4f}")


if __name__ == "__main__":
    main()
