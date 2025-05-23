import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pathlib

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()


def qrom_operation(data_dict, address_wires, value_wires):
    """
    Applies the sequence of multi-controlled X gates for a QROM.

    Args:
        data_dict (dict): Dictionary mapping address binary strings to value binary strings.
                          e.g., {'00': '101', '01': '110'}
        address_wires (list[int]): List of wire for the address.
        value_wires (list[int]): List of wire for the value.
    """
    for address_str, value_str in data_dict.items():
        # Control values are the integer representation of the address string
        control_int_list = [int(bit) for bit in address_str]

        # Determine which value qubits to flip for this address
        value_indices_to_flip = [i for i, bit in enumerate(value_str) if bit == "1"]

        # Apply a MultiControlledX for each '1' in the value string
        for value_index in value_indices_to_flip:
            # Apply the MultiControlledX to the value wire with address wires as controls
            qml.MultiControlledX(
                wires=list(address_wires) + [value_wires[value_index]],
                control_values=control_int_list,
            )


def qrom_circuit(data_dict, address_wires, input_address):
    # Prepare the address state |input_address_str>
    for wire_index, bit in enumerate(input_address):
        if bit == "1":
            qml.X(wires=address_wires[wire_index])

    # Apply the QROM operation
    qrom_operation(data_dict, address_wires, value_wires)

    # Measure probabilities
    return qml.probs(wires=list(range(num_wires)))


if __name__ == "__main__":

    # Define QROM data
    data_dict = {
        "000": "10101",
        "001": "01100",
        "010": "11011",
        "011": "00101",
        "100": "10010",
        "101": "11110",
        "110": "01001",
        "111": "10111",
    }

    # Setup Wires
    addresses = list(data_dict.keys())
    values = list(data_dict.values())
    address_length = len(addresses[0])
    value_length = len(values[0])

    num_wires = address_length + value_length

    address_wires = list(range(address_length))
    value_wires = list(range(address_length, num_wires))

    # Define circuit using the QROM operation
    @qml.qnode(qml.device("default.qubit", wires=num_wires))
    def qrom_node(input_address):
        if len(input_address) != address_length:
            raise ValueError(f"Input address '{input_address}' length mismatch.")

        return qrom_circuit(
            data_dict=data_dict,
            address_wires=address_wires,
            input_address=input_address,
        )

    # Test Execution
    test_input = "000"  # Choose an input address to test

    print(f"--- Testing QROM with Input |{test_input}> ---")
    print(f"Address wires: {address_wires}")
    print(f"Value wires:  {value_wires}")
    print(f"Data map:      {data_dict}")
    print("-" * 30)

    probabilities = qrom_node(test_input)

    # Analysis
    output_index = np.argmax(probabilities)
    output_binary = format(output_index, f"0{num_wires}b")
    output_address = output_binary[:address_length]
    output_value = output_binary[address_length:]
    expected_value = data_dict[test_input]

    # Report Results
    print(f"Input Address State:   |{test_input}>")
    print(f"Expected value State: |{expected_value}>")
    print("-" * 30)
    print(f"Measured Output State: |{output_binary}>")
    print(f" -> Address Part: |{output_address}>")
    print(f" -> value Part:  |{output_value}>")
    print(f"Probability: {probabilities[output_index]:.4f}")
    print("-" * 30)

    # Verification
    is_correct_address = output_address == test_input
    is_correct_value = output_value == expected_value
    is_high_probability = np.isclose(probabilities[output_index], 1.0)

    if is_correct_address and is_correct_value and is_high_probability:
        print("Verification Successful: QROM mapped the input correctly.")
    else:
        print("Verification Failed:")
        if not is_correct_address:
            print(
                f"  - Address mismatch: Got |{output_address}>, expected |{test_input}>"
            )
        if not is_correct_value:
            print(
                f"  - Value mismatch: Got |{output_value}>, expected |{expected_value}>"
            )
        if not is_high_probability:
            print(
                f"  - Low probability: Got {probabilities[output_index]:.4f}, expected ~1.0"
            )

    # Draw Circuit
    fig, ax = qml.draw_mpl(qrom_node)(input_address=test_input)
    fig.suptitle(f"QROM Circuit for Input |{test_input}>")
    fig.savefig(f"{SCRIPT_PATH}/visualization/qrom_circuit_{test_input}.png")
    plt.close(fig)
