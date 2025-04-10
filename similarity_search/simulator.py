"""
Quantum Simulator for Grover algorithm simulations.
"""

from math import pi, sqrt
from qiskit import transpile
from qiskit_aer import AerSimulator
from similarity_search.utils.create_state import create_state
from similarity_search.circuit_builder import QuantumCircuitBuilder


class GroverSimulationManager:
    """Class for managing Grover algorithm simulations."""

    @staticmethod
    def calculate_optimal_iterations_high(n_qubits, threshold):
        """
        Calculates the optimal number of Grover iterations for high similarity search.

        Args:
            n_qubits (int): Number of qubits
            threshold (float): Similarity threshold

        Returns:
            int: Optimal number of iterations
        """
        # Estimate fraction of states with similarity > threshold
        if threshold > 0.9:
            fraction = 0.125  # About 1/8 of states have very high similarity
        elif threshold > 0.7:
            fraction = 0.25  # About 2/8 of states have high similarity
        elif threshold > 0.5:
            fraction = 0.375  # About 3/8 of states have moderate similarity
        elif threshold > 0.3:
            fraction = 0.5  # About half of states have some similarity
        else:
            fraction = 0.75  # Most states have at least minimal similarity

        # Calculate total number of states in the system
        total_states = 2**n_qubits

        estimated_solutions = max(1, int(total_states * fraction))

        # Calculate optimal number of iterations
        # r = pi/4 * sqrt(N/M)
        optimal = int(pi / 4 * sqrt(total_states / estimated_solutions))

        return optimal

    @staticmethod
    def simulate_and_analyze(circuit, reference_state, num_qubits, threshold, shots):
        """
        Simulates the quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate
            reference_state (Statevector): Reference state
            num_qubits (int): Number of qubits
            threshold (float): Similarity threshold used in circuit construction
            shots (int): Number of simulation shots

        Returns:
            dict: Measurement results
        """
        # Use AerSimulator
        simulator = AerSimulator()

        # Transpile the circuit for the simulator
        transpiled = transpile(circuit, simulator)

        # Run the simulation
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Return the raw measurement counts
        # The similarity comparison happens entirely within the quantum circuit (Grover's oracle)
        return {
            "counts": counts,
            "total_shots": sum(counts.values())
        }

    @staticmethod
    def find_similar_states(
        reference, states_dict, num_qubits, threshold=0.1, max_results=5, shots=1024
    ):
        """
        Find quantum states similar to the reference state using quantum similarity search.

        Args:
            reference (tuple): Tuple of (index, state) where state is a Statevector
            states_dict (dict): Dictionary mapping indices to quantum states
            num_qubits (int): Number of qubits in the quantum states
            threshold (float): Similarity threshold for search
            max_results (int): Maximum number of results to return
            shots (int): Number of simulation shots

        Returns:
            list: List of (index, similarity) tuples for the most similar states
        """
        # Unpack reference tuple
        reference_idx, reference_state = reference

        # Calculate optimal iterations
        iterations = GroverSimulationManager.calculate_optimal_iterations_high(
            num_qubits, threshold
        )

        # Create the quantum circuit for similarity search
        circuit = QuantumCircuitBuilder.grover_search_high_similarity(
            reference_state, num_qubits, iterations, threshold
        )

        # Run simulation
        results = GroverSimulationManager.simulate_and_analyze(
            circuit, reference_state, num_qubits, threshold, shots
        )

        # Extract similar states directly from the quantum circuit output
        similar_states = []
        
        # Get the most frequently measured bitstrings (highest amplitude after Grover)
        # These are the ones that Grover's algorithm has amplified as similar
        sorted_counts = sorted(results["counts"].items(), key=lambda x: x[1], reverse=True)
        
        # Convert counts to indices directly
        # For each measurement result, we assign an article index
        # This approach assumes the bitstring directly maps to an index
        count = 0
        for bitstring, frequency in sorted_counts:
            # Skip the reference article
            index = int(bitstring, 2)  # Convert binary string to integer
            if index == reference_idx or index >= len(states_dict):
                continue
                
            # Calculate probability based on measurement frequency
            probability = frequency / results["total_shots"]
            similar_states.append((index, probability))
            
            count += 1
            if count >= max_results:
                break
                
        # Return whatever we found (might be fewer than max_results)
        return similar_states
