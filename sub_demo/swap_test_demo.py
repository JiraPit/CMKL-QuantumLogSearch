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
from similarity_search.circuit_builder import QuantumCircuitBuilder


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
    simulator = AerSimulator()
    shots = 1024

    # Perform swap tests
    num_tests = 10
    for i in range(num_tests):
        # Randomly select two article indices
        article_indices = list(article_states.keys())
        idx1, idx2 = random.sample(article_indices, 2)

        # Get the quantum states
        state1 = article_states[idx1]
        state2 = article_states[idx2]

        # Or calculate classical fidelity to compare with quantum result
        classical_fidelity = np.abs(np.dot(state1, np.conj(state2))) ** 2
        print(f"Classical fidelity: {classical_fidelity:.4f}")

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
        qc = QuantumCircuitBuilder.create_circuit([state1_reg, state2_reg])
        qc.initialize(state1, state1_reg)
        qc.initialize(state2, state2_reg)
        qc.add_register(ancilla_reg, cr)

        # Build swap test circuit
        swap_test = QuantumCircuitBuilder.swap_test_circuit(
            state1_reg,
            state2_reg,
            ancilla_reg,
        )
        qc = qc.compose(swap_test)
        assert isinstance(qc, QuantumCircuit)

        # Measure ancilla qubit to classical bit
        qc.measure(ancilla_reg[0], cr[0])

        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        print("Measurement results:", counts)

        # Calculate probability of measuring |0⟩
        p_zero = counts.get("0", 0) / shots

        # For SWAP test, similarity = 2*p_zero - 1
        similarity = 2 * p_zero - 1

        print(f"Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
