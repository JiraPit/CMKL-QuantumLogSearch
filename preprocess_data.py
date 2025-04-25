import pandas as pd
import argparse
import os

from hash_module import hash


def hash_database(db_folder_path, hash_length):
    """
    Reads log data from the specified folder path, hashes values column by column,
    and saves the hashed data to the same folder path.

    Args:
        db_folder_path (str): The path to the folder containing the database files.
    """
    # Construct the database paths using the provided folder path argument
    log_database_path = os.path.join(db_folder_path, "log_data.csv")
    hash_database_path = os.path.join(db_folder_path, "hash_data.csv")

    # Load the CSV file using the constructed path
    df = pd.read_csv(log_database_path, dtype=str)
    cols = df.columns.tolist()

    # For each column, generate a hash column in the hash database
    hash_df = pd.DataFrame()
    for col in cols:
        hash_df[col] = [
            "".join(hash.hash_string(str(value))[:hash_length]) for value in df[col]
        ]

    # Save the modified DataFrame to a new CSV file using the constructed path
    hash_df.to_csv(hash_database_path, index=False)


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
    hash_database(args.db, args.len)
