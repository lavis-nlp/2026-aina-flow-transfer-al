from typing import List, Union
import numpy as np


def hot_encoding_to_ints(hot_encoded_array: np.ndarray) -> List[int]:
    """
    Convert a hot-encoded 1D array to integers.

    Args:
        hot_encoded_array (np.ndarray): A 1D numpy array where each element is either 0 or 1.

    Returns:
        List[int]: A list of integers representing the indices of the 1s in the hot-encoded array.
    """
    return np.where(hot_encoded_array > 0)[0].tolist()


def convert_to_primitives_nested(obj: list | dict | np.ndarray | np.number) -> list | dict:
    """
    Convert numpy arrays in a nested structure (list or dict) to Python primitives.

    Args:
        obj (list | dict | np.ndarray): The input object which can be a list, dict, or numpy array.

    Returns:
        list | dict: The input object with numpy arrays converted to Python primitives.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_primitives_nested(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_primitives_nested(value) for key, value in obj.items()}
    elif isinstance(obj, np.number):
        return obj.item()
    else:
        return obj  # Return as is if it's neither a list, dict, nor numpy array
