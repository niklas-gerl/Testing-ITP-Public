"""
This file contains methods to create fitness functions.
"""

# 2d normal fitness functions

import secrets
from scipy.stats import multivariate_normal
import numpy as np

#Utils

def get_fitness_array(x: int, y: int, fct: callable, scaling_factor: float = 1) -> np.ndarray:
    """
    Turns a callable fitness function into an array containing the fitness.
    Normalizes all fitness values to values between 0 and 1 and multiplies them with scaling_factor
    :param x: the length of x coordinate
    :param y: the length of y coordinate
    :param fct: the fitness function
    :return: the results of the fitness function within x*y as an array
    """
    fitness_array = np.empty(shape=(x, y))
    for i in range(x):
        for j in range(y):
            fitness_array[i][j] = fct(i, j)

    min = np.min(fitness_array)
    max = np.max(fitness_array)
    fitness_array = (fitness_array - min) / (max - min)

    fitness_array = np.round(fitness_array, 3)

    return scaling_factor * fitness_array



# 2d normal fitness fcts for first experiment set

# centered mean wih different covariances
normal_1 = multivariate_normal(mean=[5, 5], cov=[[1, 0.5], [0.5, 1]])
normal_1_arr = get_fitness_array(10, 10, lambda x, y: normal_1.pdf([x, y]), scaling_factor=100)

normal_2 = multivariate_normal(mean=[5, 5], cov=[[3, 1.5], [1.5, 3]])
normal_2_arr = get_fitness_array(10, 10, lambda x, y: normal_2.pdf([x, y]), scaling_factor=100)

normal_3 = multivariate_normal(mean=[5, 5], cov=[[5, 3], [3, 5]])
normal_3_arr = get_fitness_array(10, 10, lambda x, y: normal_3.pdf([x, y]), scaling_factor=100)


# same cov but different placed mean
normal_4 = multivariate_normal(mean=[2, 2], cov=[[5, 3], [3, 5]])
normal_4_arr = get_fitness_array(10, 10, lambda x, y: normal_4.pdf([x, y]), scaling_factor=100)

normal_5 = multivariate_normal(mean=[2, 7], cov=[[5, 3], [3, 5]])
normal_5_arr = get_fitness_array(10, 10, lambda x, y: normal_5.pdf([x, y]), scaling_factor=100)

normal_6 = multivariate_normal(mean=[7, 2], cov=[[5, 3], [3, 5]])
normal_6_arr = get_fitness_array(10, 10, lambda x, y: normal_6.pdf([x, y]), scaling_factor=100)

normal_7 = multivariate_normal(mean=[7, 7], cov=[[5, 3], [3, 5]])
normal_7_arr = get_fitness_array(10, 10, lambda x, y: normal_7.pdf([x, y]), scaling_factor=100)

# added multivar normal
# summed multivar normal
normal_8_arr = normal_4_arr + normal_7_arr
normal_9_arr = normal_5_arr + normal_6_arr
normal_10_arr = normal_8_arr + normal_9_arr


def normal_x_linear_y(x, y):
    """
    A fitness function that is monotonically increasing along y-axis and normal along x-axis
    """
    return x * [0.013, 0.443, 5.399, 24.197, 24.197, 39.894, 24.197, 5.399, 0.443, 0.013][y]

def get_random_2d_normal():
    """
    creates a 2d normal fitness function with random mean and cov.
    cov is [[cov_01_10,0],[0,cov_01_10]], where cov_01_10 is drawn from a uniform distr from 0.2 to 2.
    mean is [m1, m2] where m1 and m2 are drawn from uniform distribution from 0 to 9
    """


    rng = np.random.default_rng(secrets.randbits(128))
    cov_01_10 = rng.integers(2,20)/10

    cov = [[cov_01_10,0],[0,cov_01_10]]
    mean = rng.integers(0, 10, size = 2)

    fct = multivariate_normal(mean=mean, cov=cov)
    array = get_fitness_array(10,10, lambda x, y: fct.pdf([x,y]), scaling_factor=100)

    return fct, array

def get_n_random_2d_normal_fcts(n):
    """
    uses get_random_2d_normal() to get n random 2d normal fitness fcts.
    :param n: number of fitness functions to be drawn
    """

    res = []

    for i in range(n):
        _, array = get_random_2d_normal()
        res.append(array)

    return np.array(res)



#
normal_x_linear_y_array = get_fitness_array(10, 10, lambda x, y: normal_x_linear_y(x, y), scaling_factor=100)


def get_all_arr():
    """
    returns all previously defined fitness functions
    """
    return [normal_1_arr, normal_2_arr, normal_3_arr, normal_4_arr, normal_5_arr, normal_6_arr, normal_7_arr,
            normal_8_arr,
            normal_9_arr, normal_10_arr, normal_x_linear_y_array]


def get_n_fcts(n):
    fcts_arrays = []
    means = [[2, 5], [7, 2], [7, 7]]

    for i in range(n):
        fct = multivariate_normal(mean=means[i], cov=[[1, 0.5], [0.5, 1]])
        fcts_arrays.append(get_fitness_array(10, 10, lambda x, y: fct.pdf([x, y]), scaling_factor=100))

    return np.array(fcts_arrays)
#random 2d normal fitness functions for 2nd experiment set
def get_random_2d_normal():
    """
    creates a 2d normal fitness function with random mean and cov.
    cov is [[cov_01_10,0],[0,cov_01_10]], where cov_01_10 is drawn from a uniform distr from 0.2 to 2.
    mean is [m1, m2] where m1 and m2 are drawn from uniform distribution from 0 to 9
    """


    rng = np.random.default_rng(secrets.randbits(128))
    cov_01_10 = rng.integers(2,20)/10

    cov = [[cov_01_10,0],[0,cov_01_10]]
    mean = rng.integers(0, 10, size = 2)

    fct = multivariate_normal(mean=mean, cov=cov)
    array = get_fitness_array(10,10, lambda x, y: fct.pdf([x,y]), scaling_factor=100)

    return fct, array

def get_n_random_2d_normal_fcts(n):
    """
    uses get_random_2d_normal() to get n random 2d normal fitness fcts.
    :param n: number of fitness functions to be drawn
    """

    res = []

    for i in range(n):
        _, array = get_random_2d_normal()
        res.append(array)

    return np.array(res)