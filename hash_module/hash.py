import hashlib


def hash_string(input_string: str) -> str:
    """Hashes a string into a bitstring of 256 bits using SHA-3 256."""
    encoded_string = input_string.encode("utf-8")
    hasher = hashlib.sha3_256()
    hasher.update(encoded_string)
    hash_bytes = hasher.digest()
    binary_string = "".join(format(byte, "08b") for byte in hash_bytes)
    binary_list = [bit for bit in binary_string]
    bitstring = "".join(binary_list)
    return bitstring


if __name__ == "__main__":
    test_string = "Hello, world!"
    binary = hash_string(test_string)

    print(f"Input string: '{test_string}'")
    print(f"Binary string: {binary}")
