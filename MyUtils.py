from datetime import datetime
import os
import numpy as np
import json
import subprocess
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORDS_DIR = os.path.join(SCRIPT_DIR, 'data', 'records')
PARAMS_DIR = os.path.join(SCRIPT_DIR, 'data', 'simulation_params')

PRINT_ENABLED = True


class DataSaver:
    """A DataSavers can store data during simulations and dump them into a json file afterward. """
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        guarantee_records_directory()

        self.file_path = os.path.join(RECORDS_DIR, experiment_name + '_' + self.time_stamp + '.txt')
        with open(self.file_path, 'w') as file:
            json.dump({}, file)

        self.data = {}

    def save_meta_data(self, meta_data: dict):
        """
        Saves meta data of simulation
        :param meta_data: metdadata of simulation
        """
        self.data['meta_data'] = meta_data

    def save(self, key, value):
        """
        Saves a key value pair into dict. Possibly converts numpy arrays to lists.
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self.data[key] = value

    def dump_data(self):
        """
        Dumps the data in the dict into a file.
        """
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file)


def guarantee_records_directory() -> None:
    """creates the records directory if necessary"""
    if not os.path.exists(RECORDS_DIR):
        os.makedirs(RECORDS_DIR)


def my_print(message) -> None:
    """
    Prints the message only if PRINT_ENABLED is True
    :param message: message to print
    """
    if PRINT_ENABLED:
        print(message)