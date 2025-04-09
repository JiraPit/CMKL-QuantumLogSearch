"""
Quantum Simulator for Grover algorithm simulations.
"""

from math import pi, sqrt
from qiskit import transpile
from qiskit_aer import AerSimulator

# Import QuantumState from state module
from state.quantum_state import QuantumState


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

        # For small number of qubits, limit iterations to avoid overshooting
        return min(2, max(1, optimal))

    @staticmethod
    def simulate_and_analyze(circuit, reference_state, num_qubits, threshold, shots):
        """
        Simulates the quantum circuit and analyzes results.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate
            reference_state (Statevector): Reference state for similarity comparison
            num_qubits (int): Number of qubits
            threshold (float): Similarity threshold used
            shots (int): Number of simulation shots

        Returns:
            dict: Measurement results and statistics
        """
        # Use AerSimulator (modern Qiskit approach)
        simulator = AerSimulator()

        # Transpile the circuit for the simulator
        transpiled = transpile(circuit, simulator)

        # Run the simulation
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Analyze similarities for each measured state
        similarities = {}
        for bitstring, count in counts.items():
            # Ensure consistent bit ordering
            if len(bitstring) != num_qubits:
                bitstring = bitstring.zfill(num_qubits)

            # Create state vector for this bitstring
            state = QuantumState.create_target_state(num_qubits, bitstring)

            # Calculate similarity to reference state
            sim = abs(state.inner(reference_state)) ** 2

            # Store result
            similarities[bitstring] = {
                "count": count,
                "similarity": sim,
                "above_threshold": sim >= threshold,
            }

        # Calculate statistics
        total_shots = sum(item["count"] for item in similarities.values())
        above_threshold_shots = sum(
            item["count"] for item in similarities.values() if item["above_threshold"]
        )
        success_rate = above_threshold_shots / total_shots if total_shots > 0 else 0

        # Sort results by similarity
        sorted_results = sorted(
            similarities.items(), key=lambda x: x[1]["similarity"], reverse=True
        )

        return {
            "counts": counts,
            "similarities": similarities,
            "sorted_results": sorted_results,
            "success_rate": success_rate,
            "above_threshold_count": above_threshold_shots,
            "total_shots": total_shots,
        }

