"""
Quantum Simulator for Grover algorithm simulations.
"""

from math import pi, sqrt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
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
        # Calculate total number of states in the system
        total_states = 2**n_qubits

        # Calculate the expected fraction of states with similarity > threshold
        # For quantum states, similarity relates to the inner product
        # The threshold relates to the overlap probability, which is related to
        # the distribution of inner product squared between random quantum states

        # For random unit vectors in high dimensions, the distribution of inner products
        # follows approximately a normal distribution, centered at 0 with standard deviation 1/sqrt(d)
        # where d is the dimension (2^n_qubits)
        # The probability of overlap > threshold decreases exponentially with threshold

        # We use an exponential model: fraction = a * exp(-b * threshold)
        # Where a and b are constants chosen to approximate the distribution
        a = 0.9  # Maximum fraction at threshold = 0
        b = 3.0  # Rate of decay with increasing threshold

        fraction = a * np.exp(-b * threshold)
        # Clamp to reasonable values
        fraction = max(0.01, min(0.99, fraction))  # At least 1% and at most 99%

        estimated_solutions = max(1, int(total_states * fraction))

        # Calculate optimal number of iterations
        # r = pi/4 * sqrt(N/M)
        optimal = int(pi / 4 * sqrt(total_states / estimated_solutions))

        # Ensure at least one iteration
        return max(1, optimal)

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
        return {"counts": counts, "total_shots": sum(counts.values())}

    @staticmethod
    def find_similar_states(
        reference, states_dict, num_qubits, threshold, max_results, shots
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
        sorted_counts = sorted(
            results["counts"].items(), key=lambda x: x[1], reverse=True
        )

        # Convert counts to indices directly
        # For each measurement result, we assign an article index
        # This approach assumes the bitstring directly maps to an index
        count = 0
        for bitstring, frequency in sorted_counts:
            index = int(bitstring, 2)  # Convert binary string to integer

            # TODO: Skip the reference article
            # if index == reference_idx:
            #     continue

            if index not in states_dict.keys():
                continue

            # Calculate probability based on measurement frequency
            probability = frequency / results["total_shots"]
            similar_states.append((index, probability))

            count += 1
            if count >= max_results:
                break

        # Return whatever we found (might be fewer than max_results)
        return similar_states
