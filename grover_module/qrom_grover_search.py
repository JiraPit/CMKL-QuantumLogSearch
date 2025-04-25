import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import math
import pathlib
from qrom_module import qrom

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()


def multi_controlled_Z(wires: list[int]):
    """Applies a multi-controlled Z gate using H-MCX-H decomposition.

    Args:
        wires (list[int]): List of wires to apply the MCZ gate on.
                            The last wire is the target for the internal MCX.
    """
    if len(wires) == 1:
        qml.PauliZ(wires=wires[0])  # Just apply a Z gate for 1 qubit
        return

    # Apply H to target wire (last wire)
    qml.Hadamard(wires=wires[-1])

    # Apply MCX to the target wire (last wire) using all the other wires as controls
    control_values = [1] * len(wires[:-1])
    qml.MultiControlledX(wires=wires, control_values=control_values)

    # Apply H to target wire again
    qml.Hadamard(wires=wires[-1])


def oracle(
    qrom_data: dict,
    target_value: str,
    address_wires: list[int],
    value_wires: list[int],
):
    """Applies the Oracle operation using QROM for Grover's search.

    Marks the address states |a> such that qrom_data[a] == target_value.

    Args:
        qrom_data (dict): The dictionary defining the QROM mapping.
        target_value (str): The binary string value to search for in the QROM output.
        address_wires (list[int]): Wires for the QROM addresses (search space for grover).
        value_wires (list[int]): Wires for the QROM values.
    """
    if len(target_value) != len(value_wires):
        raise ValueError(
            f"Length of search_target_value ('{target_value}', length {len(target_value)}) doesn't match the QROM data length ({len(qrom_data)})."
        )

    # Apply QROM: |a>|0> -> |a>|V(a)>
    qrom.qrom_operation(qrom_data, address_wires, value_wires)

    # Apply X gates to target wires to transform the desired state |target_value> to |11...1>
    for i, bit in enumerate(target_value):
        if bit == "0":
            qml.PauliX(wires=value_wires[i])

    # Mark the state where target wires are |11...1> using MCZ
    # This flips the phase of the state |a>|1...1> (originally |a>|target_value>)
    multi_controlled_Z(value_wires)

    # Uncompute the X gates
    for i, bit in enumerate(target_value):
        if bit == "0":
            qml.PauliX(wires=value_wires[i])

    # Uncompute QROM: |a>|V(a)> -> |a>|0>
    # Since QROM uses MCX gates, its inverse is itself.
    qrom.qrom_operation(qrom_data, address_wires, value_wires)


def diffusion(wires: list[int]):
    """Applies the Grover Diffusion operator (amplitude amplification).

    Args:
        wires (list[int]): List of wires representing the search space.
    """
    # Apply H to each wire
    for w in wires:
        qml.Hadamard(wires=w)

    # Apply X to each wire
    for w in wires:
        qml.PauliX(wires=w)

    # Apply MCZ (controlled by all wires)
    multi_controlled_Z(wires)

    # Apply X to each wire
    for w in wires:
        qml.PauliX(wires=w)

    # Apply H to each wire
    for w in wires:
        qml.Hadamard(wires=w)


def grover_circuit(
    qrom_data: dict,
    target_value: str,
    address_wires: list[int],
    value_wires: list[int],
    num_iterations: int,
):
    """Defines the Grover circuit with QROM.
    Args:
        qrom_data (dict): The dictionary defining the QROM mapping.
        target_value (str): The binary string value to search for in the QROM output.
        address_wires (list[int]): Wires for the QROM addresses (search space for grover).
        value_wires (list[int]): Wires for the QROM values.
    """
    # Initialize address wires to uniform superposition
    for w in address_wires:
        qml.Hadamard(wires=w)

    # Apply Grover iterations
    for _ in range(num_iterations):
        oracle(qrom_data, target_value, address_wires, value_wires)
        diffusion(address_wires)

    # Measure probabilities of the address wires
    return qml.probs(wires=address_wires)


if __name__ == "__main__":

    # Define QROM data
    data_dict = {
        "0000": "10101",
        "0001": "01100",
        "0010": "11011",
        "0011": "00101",
        "0100": "10010",
        "0101": "11110",
        "0110": "01001",
        "0111": "10111",
        "1000": "00011",
        "1001": "11000",
        "1010": "01111",
        "1011": "10001",
        "1100": "00110",
        "1101": "11101",
        "1110": "01011",
        "1111": "00000",
    }

    # Define target value
    target_value = "00011"

    print("--- Grover Search with QROM ---")
    print(f"Searching for address 'a' such that QROM(a) = |{target_value}>")
    print(f"QROM Data: {data_dict}")

    # Setup Wires
    addresses = list(data_dict.keys())
    values = list(data_dict.values())

    address_length = len(addresses[0])
    value_length = len(values[0])

    num_wires = address_length + value_length

    address_wires = list(range(address_length))
    target_wires = list(range(address_length, num_wires))

    print("\nWire Configuration:")
    print(f"  Address Wires: {address_wires} ({address_length} qubits)")
    print(f"  Value Wires: {target_wires} ({value_length} qubits)")
    print(f"  Total Wires: {num_wires}")

    # Find Solutions and Calculate Iterations
    # *** This solution is ONLY used in the result analysis, NOT in the circuit ***
    solutions = [addr for addr, value in data_dict.items() if value == target_value]
    num_solutions = len(solutions)
    search_space_size = 2**address_length

    # Calculate the optimal number of iterations for Grover's algorithm
    if num_solutions == 0:
        print(
            f"\nWarning: The target value '{target_value}' is not present in the QROM data."
        )
        num_iterations = 1  # Still run once to see the ~uniform distribution
    else:
        num_iterations = math.floor(
            np.pi / 4 * np.sqrt(search_space_size / num_solutions)
        )
        print(f"Search space size N = 2^{address_length} = {search_space_size}")
        print(f"Optimal number of Grover iterations: {num_iterations}")

    # Define the Grover circuit
    @qml.qnode(qml.device("default.qubit", wires=num_wires))
    def grover_node():
        return grover_circuit(
            qrom_data=data_dict,
            target_value=target_value,
            address_wires=address_wires,
            value_wires=target_wires,
            num_iterations=num_iterations,
        )

    # Execute the Circuit
    probabilities = grover_node()

    print("\n--- Results ---")

    # Find the index with the highest probability
    most_likely_index = np.argmax(probabilities)

    # Convert index to binary address string
    measured_address = format(most_likely_index, f"0{address_length}b")
    measured_prob = probabilities[most_likely_index]

    print(f"Most likely address index: {most_likely_index}")
    print(f"Measured Address State: |{measured_address}>")
    print(f"Probability: {measured_prob:.4f}")

    # Verification
    print("\n--- Verification ---")
    if measured_address in solutions:
        print(f"Success! Measured address '{measured_address}' is a correct solution.")
        print(
            f"   QROM('{measured_address}') = {data_dict[measured_address]}, matches target '{target_value}'."
        )
    else:
        print(f"Failure! Measured address '{measured_address}' is incorrect.")
        print(f"   Expected one of: {solutions}")

    # Plot probabilities
    plt.figure(figsize=(10, 6))
    plt.bar(range(search_space_size), probabilities)
    plt.xlabel("Address State Index (Decimal)")
    plt.ylabel("Probability")
    plt.title(f"Grover Search Results (Target Value: {target_value})")

    # Highlight the solution(s)
    solution_indices = [int(s, 2) for s in solutions]
    for idx in solution_indices:
        if idx < len(probabilities):  # Check bounds
            plt.bar(
                idx,
                probabilities[idx],
                color="red",
                label=(
                    f'Solution: |{format(idx, f"0{address_length}b")}>'
                    if idx == solution_indices[0]
                    else ""
                ),
            )
    if solution_indices:
        plt.legend()
    plt.xticks(
        ticks=range(search_space_size),
        labels=[format(i, f"0{address_length}b") for i in range(search_space_size)],
        rotation=90,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(f"{SCRIPT_PATH}/visualization/grover_qrom_probs_{target_value}.png")
    print(
        f"\nProbability plot saved as visualization/grover_qrom_probs_{target_value}.png"
    )

    # Draw Circuit
    if num_iterations <= 3:  # Only draw if few iterations
        fig, ax = qml.draw_mpl(grover_node)()
        fig.suptitle(f"Grover Circuit with QROM Oracle (Target: {target_value})")
        fig.savefig(
            f"{SCRIPT_PATH}/visualization/grover_qrom_circuit_{target_value}.png"
        )
        print(
            f"Circuit diagram saved as visualization/grover_qrom_circuit_{target_value}.png"
        )
    else:
        print("\nSkipping circuit drawing (circuit is large due to many iterations).")
