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
    value_length = len(value_wires)

    # Loop through each value bit index
    for value_index in range(value_length):
        # Go through each address that sets this value bit to 1
        for address_str, value_str in data_dict.items():
            if value_str[value_index] == "1":
                control_bits = [int(b) for b in address_str]

                # Use control_values to specify the control bits for the value bit
                qml.MultiControlledX(
                    wires=list(address_wires) + [value_wires[value_index]],
                    control_values=control_bits,
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
    # data_dict = {
    #     "0000": "10101",
    #     "0001": "01100",
    #     "0010": "11011",
    #     "0011": "00101",
    #     "0100": "10010",
    #     "0101": "11110",
    #     "0110": "01001",
    #     "0111": "10111",
    #     "1000": "00011",
    #     "1001": "11000",
    #     "1010": "01111",
    #     "1011": "10001",
    #     "1100": "00110",
    #     "1101": "11101",
    #     "1110": "01011",
    #     "1111": "00000",
    # }
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
    # test_input = "0000"
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
    fig.savefig(f"{SCRIPT_PATH}/visualization/optimized_qrom_circuit_{test_input}.png")
    plt.close(fig)
