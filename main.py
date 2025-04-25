import pandas as pd
import pennylane as qml
import numpy as np
import argparse

from hash_module import hash
from grover_module import qrom_grover_search


def main(database_dir, hash_length):
    log_database = f"{database_dir}/log_data.csv"
    hash_database = f"{database_dir}/hash_data.csv"
    df = pd.read_csv(log_database, dtype=str)
    hash_df = pd.read_csv(hash_database, dtype=str)

    while True:
        print("\n--- Command Menu ---")
        print("1. Search for a log entry")
        print(": search <field>=<value>")
        print("2. Add a log entry")
        print(": add <field1>=<value1>,<field2>=<value2>,...")
        print("3. Show last 5 log entries")
        print(": show")
        print()

        command = input("Enter a command:")
        try:
            command, command_args = command.strip().split(" ")
        except Exception:
            command_args = ""

        if command == "search":
            handle_search(command_args, df, hash_df, hash_length)
        elif command == "add":
            df, hash_df = handle_add(command_args, df, hash_df, hash_length)
        elif command == "show":
            handle_show(df, hash_df)


def handle_search(command_args, df, hash_df, hash_length):
    """Search for a log entry based on a specific field and value."""
    search_field, search_value = command_args.split("=")
    if search_field not in df.columns:
        print(f"Field '{search_field}' does not exist.")
        return

    # Calculate lengh of address and value in bits
    address_length = len(f"{len(hash_df[search_field]):b}")
    value_length = len(hash_df[search_field][0])

    print(f"Address length: {address_length} bits")
    print(f"Value length: {value_length} bits")

    # Construct the data dictionary for QROM
    data_dict = {
        f"{i:0{address_length}b}": value
        for i, value in enumerate(hash_df[search_field])
    }

    # Setup Wires
    num_wires = address_length + value_length
    address_wires = list(range(address_length))
    target_wires = list(range(address_length, num_wires))

    # Calculate the optimal number of iterations for Grover's algorithm
    num_iterations = int(np.floor(np.pi / 4 * np.sqrt(len(data_dict))))

    # Hash the value
    hashed_search_value = hash.hash_string(search_value)[:hash_length]

    # Define the Grover circuit
    @qml.qnode(qml.device("lightning.qubit", wires=num_wires))
    def grover_node():
        return qrom_grover_search.grover_circuit(
            qrom_data=data_dict,
            target_value=hashed_search_value,
            address_wires=address_wires,
            value_wires=target_wires,
            num_iterations=num_iterations,
        )

    # Search for the hashed value in the DataFrame
    probabilities = grover_node()

    # Find all indices messured with enough probability
    max_prob = np.max(probabilities)
    threshold = max_prob * 0.9
    found_indices = np.where(probabilities >= threshold)[0]

    # Convert index to binary address string
    for found_index in found_indices:
        measured_prob = probabilities[found_index]
        measured_address = format(found_index, f"0{address_length}b")
        print(
            f"\nMeasured Output Address State: |{measured_address}> which is index {found_index} with probability {measured_prob:.4f}"
        )
        print(df.loc[found_index])
        print()


def handle_add(command_args, df, hash_df, hash_length):
    """Add a new log entry to the DataFrame and update the hash DataFrame."""
    new_entry = {}
    for field in command_args.split(","):
        key, value = field.split("=")
        if key not in df.columns:
            print(f"Field '{key}' does not exist.")
            return df, hash_df
        new_entry[key] = value

    # Append the new entry to the DataFrame
    df = pd.concat(
        [df, pd.DataFrame([new_entry], columns=df.columns)], ignore_index=True
    )

    # Hash the new entry and append to hash_df
    hashed_entry = {
        key: hash.hash_string(value)[:hash_length] for key, value in new_entry.items()
    }
    hash_df = pd.concat(
        [hash_df, pd.DataFrame([hashed_entry], columns=hash_df.columns)],
        ignore_index=True,
    )
    print("New entry added successfully.")
    print()

    return df, hash_df


def handle_show(df, hash_df):
    """Display the last 5 log entries and their hashed values."""
    print("\nLast 5 log entries:")
    print(df.tail(5))
    print("\nLast 5 hashed log entries:")
    print(hash_df.tail(5))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hash data from log_data.csv and save to hash_data.csv within a specified directory."
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the folder containing log_data.csv and where hash_data.csv will be saved.",
    )
    parser.add_argument(
        "--len",
        type=int,
        default=12,
        help="Length of the hash to be generated.",
    )
    args = parser.parse_args()

    main(args.db, args.len)
