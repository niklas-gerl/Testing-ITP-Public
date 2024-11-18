"""
This module provides functions for creating, expanding and printing worlds.
"""

import numpy as np

def create_world(resource_min: int, resource_max: int, world_width: int = 10, world_height: int = 10):
    """creates a 10x10 world which consists of a 10x10 grid with random numbers from ressource_min to ressource_max """
    return np.random.randint(resource_min, resource_max + 1, size=(world_width, world_height, 2))


def expand_world(orig_world, resource_min: int, resource_max: int, expansion_factor: int):
    """
    takes an existing world and expands it at the edges of the world
    expanding the world with one call and expansion_size n is significantly faster
    than calling it n times with expansion size=1

    :param orig_world: the world to expand
    :param resource_min: the minimum quantity of a ressource (included)
    :param resource_max: the maximum quantity of a ressource (included)
    :param expansion_factor: the number of rings of cells that get added to the world
    :return: the new world as np array
    """
    width, height, _ = orig_world.shape

    new_world = np.random.randint(resource_min, resource_max + 1,
                                  size=(width + expansion_factor * 2, height + expansion_factor * 2, 2))
    new_world[expansion_factor:width + expansion_factor, expansion_factor:height + expansion_factor] = orig_world

    return new_world


def print_world(world_to_print) -> None:
    """
    prints a world in a readable format
    """
    for row in world_to_print:
        print('')
        for vec in row:
            print(vec, end='\t')