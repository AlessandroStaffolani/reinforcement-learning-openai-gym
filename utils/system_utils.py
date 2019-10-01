import os


def get_path(path):
    os.makedirs(path, exist_ok=True)
    return path
