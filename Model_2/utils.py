import os
import numpy as np


class ArgumentParser:
    def __init__(self, dict):
        self.__dict__.update(dict)


def load_data(path):
    files = os.listdir(path)
    data = []
    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            data.append(f.read())
    return data


def get_vocabulary(data):
    chars = sorted(list(set(data)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char
