import GeneticAlgorithm as GA
from concurrent.futures import ProcessPoolExecutor
import FitnessFunctions


def repeat_simulations_helper(i, no_fitness_functions):
    """
    wrapper class for simulation used by class for multiprocessing multiple simulations.
    Handles the input from executor.map() call.
    Handles possible exceptions from simulation() and makes them visible
    """
    try:
        fitness_arrays = FitnessFunctions.get_n_random_2d_normal_fcts(no_fitness_functions)
        GA.simulate(result_file_prefix=str(i) + '-' + 'NOF' + str(no_fitness_functions) + GA.RESULT_FILE_PREFIX, fitness_fcts=fitness_arrays)
    except Exception as e:
        print(e)


def repeat_simulations(number_repetitions: int, nos_fitness_functions) -> None:
    """
    Repeats the current standard simulation multiple times using multiprocessing.
    Output files while most likely have the same time stamps, though they have an id before the timestamp.
    Each simulation will draw a new set of random fitness functions.
    :param number_repetitions: how often each number of fitness functions gets repeated
    :param nos_fitness_functions: The numbers of fitness functions to draw and use for experiments
    """

    for no_fitness_functions in nos_fitness_functions:
        with ProcessPoolExecutor() as executor:
            executor.map(repeat_simulations_helper, range(number_repetitions), [no_fitness_functions] * number_repetitions)


def multiple_simulations(parameter_sets: list[dict]):
    """
    start multiple simulations with different parameters in parallel
    """

    with ProcessPoolExecutor() as executor:
        for parameter_set in parameter_sets:
            executor.submit(GA.simulate,
                            parameter_set['no_agents_per_generation'],
                            parameter_set['no_explorations_per_fitness_estimation'],
                            parameter_set['no_actions_per_exploration'],
                            parameter_set['tournament_size'],
                            parameter_set['mutation_probability'],
                            parameter_set['no_generations'],
                            parameter_set['result_file_prefix'],
                            parameter_set['fitness_fcts']
                            )


def test_simulation_runtime(save_results: bool = False, result_file_prefix=''):
    """
    calls the simulation method with low parameters to decease loop size and spped up the simulation for debugging.
    """
    SAVE_RESULTS = save_results

    GA.simulate(no_explorations_per_fitness_estimation=20,
                no_actions_per_exploration=30,
                tournament_size=5,
                mutation_probability=0.01,
                no_generations=10,
                result_file_prefix=result_file_prefix,
                print_enabled=True
                )

