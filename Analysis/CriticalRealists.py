"""
File to get some critical realist perception strategies and also decision genomes (is used for plotting only, not for simulations)
"""
import numpy as np


def get_critical_realist_x(border: int):
    """creates a critical realist with homomorphism on the x-property"""
    perception = np.zeros(shape=(10, 10), dtype=int)
    for i in range(10):
        perception[i][0:border] = 1

    return perception.flatten()


def get_critical_realist_y(border: int):
    """creates a critical realist with homomorphism on the y-property"""

    perception = np.zeros(shape=(10, 10), dtype=int)
    for i in range(border):
        perception[i] = 1
    return perception.flatten()


def get_critical_realist_xy(border: int):
    """creates a critical realist with homomorphism on the sum of x- and y-property"""

    perception = np.zeros(shape=(10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            if i + j > border:
                perception[i][j] = 1
    return perception.flatten()


def decision_prefer_x(preference: str):
    """creates and returns a decision genome of length 32 that prefers category x"""
    decision = np.zeros(shape=32, dtype=int)
    for i in range(32):
        base_2_rep = np.base_repr(i, 2)
        while len(base_2_rep) < 5:
            base_2_rep = '0' + base_2_rep
        for char_index in range(5):
            if base_2_rep[char_index] == preference:
                decision[i] = char_index
                break
    return decision
