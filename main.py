"""
Quantum Similarity Search Implementation

A quantum algorithm that implements a multi-solution quantum search with an oracle
that checks if cosine similarity between a state and a target exceeds a threshold.

Requirements:
- Qiskit
- Qiskit-Aer
- NumPy
- Math
"""

# Import required modules
from state.quantum_state import QuantumState
from similarity_search.circuit_builder import QuantumCircuitBuilder
from similarity_search.simulator import GroverSimulationManager
from embedding.sentence_embedding import SentenceEmbedder

# Global Configuration
DEFAULT_CONFIG = {
    "NUM_QUBITS": 5,  # Number of qubits to use in quantum circuits
    "THRESHOLD": 0.1,  # Similarity threshold for search
    "SHOTS": 1024,  # Number of simulation shots
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",  # Default sentence embedding model
}


class QuantumRecommender:
    """Main class for quantum similarity search system."""

    def __init__(self, config=None):
        """
        Initialize the quantum recommender system with configuration.

        Args:
            config (dict): Configuration dictionary with parameters
        """
        self.config = config or DEFAULT_CONFIG
        self.sentence_embedder = SentenceEmbedder(model=self.config["EMBEDDING_MODEL"])

    def run_high_similarity_search(self, target_state_input=None):
        """
        Runs the complete Grover search for states with high similarity to target.

        Args:
            target_state_input: Input target state (bitstring, amplitude list, or "random")

        Returns:
            dict: Analysis results
        """
        num_qubits = self.config["NUM_QUBITS"]
        threshold = self.config["THRESHOLD"]
        shots = self.config["SHOTS"]

        # Create or use provided target state
        if target_state_input is None:
            target_state = QuantumState.create_target_state(num_qubits, "random")
        else:
            target_state = QuantumState.create_target_state(
                num_qubits, target_state_input
            )

        # Calculate optimal iterations
        iterations = GroverSimulationManager.calculate_optimal_iterations_high(
            num_qubits, threshold
        )

        # Create the circuit
        circuit = QuantumCircuitBuilder.grover_search_high_similarity(
            target_state, num_qubits, iterations, threshold
        )

        # Run simulation
        results = GroverSimulationManager.simulate_and_analyze(
            circuit, target_state, num_qubits, threshold, shots
        )

        return results

    def embed_sentences(self, sentences):
        """
        Embed a list of sentences using the sentence embedder.

        Args:
            sentences (list): List of sentences to embed

        Returns:
            numpy.ndarray: Array of sentence embeddings
        """
        return self.sentence_embedder.embed(sentences)


# Example usage
if __name__ == "__main__":
    # Create the quantum recommender system with default config
    recommender = QuantumRecommender()

    # Run with random target state
    results = recommender.run_high_similarity_search()

    # Test sentence embedding
    test_sentences = [
        "Quantum Computing Cookbook",
        "How to bake a cake",
    ]
    embeddings = recommender.embed_sentences(test_sentences)

    # Print basic results
    print("\nSimulation Results:")
    print(f"Total shots: {results['total_shots']}")
    print(
        f"Shots with similarity ≥ {recommender.config['THRESHOLD']}: {results['above_threshold_count']}"
    )
    print(f"Success rate: {results['success_rate']*100:.2f}%")

    print(
        f"\nGenerated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
    )
    print(f"First embedding: {embeddings[0][:5]}...")
