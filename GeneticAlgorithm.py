"""
This file is for doing simulations with a static fitness function.
The simulation() method can be called directly in the main method or via other methods, e.g. to run multiple simulations
in parallel or using specific (smaller) parameters for testing purposes.
"""
import secrets
import numpy as np

import FitnessFunctions
import MyUtils
from MyUtils import DataSaver, my_print
import World
import datetime

from numba import njit

start_time = datetime.datetime.now()

# parameters


# data saving

# whether the results should be saved in a file. Can be set to False while debugging to avoid unnecessary file creation
SAVE_RESULTS = True

# The prefix of the file where results will be saved
RESULT_FILE_PREFIX = 'GA_results'

# world and agents

# minimum and maximum resource quantity boundaries (both included)
RESOURCE_MIN = 0
RESOURCE_MAX = 9

# rate at which world is expanded when agent arrives at edge. Only influences runtime performance, not results
WORLD_EXPANSION_SIZE = 3

# How much fitness points it costs the agent to take a step
DEFAULT_STEP_COSTS = 55

# number of perceptual categories (e.g. colors)
NO_PERCEPTUAL_CATEGORIES = 4

PERCEPTION_GENOME_LENGTH = 100

DEFAULT_DECISION_GENOME_LENGTH = NO_PERCEPTUAL_CATEGORIES ** 5

# number of different actions that can be performed
NO_ACTIONS = 5


# genetic algorithm##
DEFAULT_NO_AGENTS_PER_GENERATION = 200
DEFAULT_NO_EXPLORATIONS_PER_FITNESS_ESTIMATION = 100
DEFAULT_NO_ACTIONS_PER_EXPLORATION = 100
DEFAULT_TOURNAMENT_SIZE = 15
DEFAULT_MUTATION_PROBABILITY = 0.01
DEFAULT_NO_GENERATIONS = 500
DEFAULT_RESULT_FILE_PREFIX = RESULT_FILE_PREFIX

# fitness functions
DEFAULT_NO_AGENT_STATES = 1
DEFAULT_FITNESS_ARRAYS = FitnessFunctions.get_n_fcts(DEFAULT_NO_AGENT_STATES)


@njit
def perceive(world, position_row, position_col, agent_perception) -> str:
    """
    simulates perception of a given agent in a specific world
    :param world: the world of the agent
    :param position_row: the row of the agent
    :param position_col: the col of the agent
    :param agent_perception: the agent's perception genome
    :return: A string containing 5 numbers, each representing the perception of the current, left, right, up, down fields.
    """
    perception_str = ''

    # current
    world_state = world[position_row][position_col]
    perception_str += str(agent_perception[world_state[0] * 10 + world_state[1]])

    # left
    world_state = world[position_row][position_col - 1]
    perception_index = world_state[0] * 10 + world_state[1]
    perception_str += str(agent_perception[perception_index])

    # right
    world_state = world[position_row][position_col + 1]
    perception_index = world_state[0] * 10 + world_state[1]
    perception_str += str(agent_perception[perception_index])

    # up
    world_state = world[position_row - 1][position_col]
    perception_index = world_state[0] * 10 + world_state[1]
    perception_str += str(agent_perception[perception_index])

    # down
    world_state = world[position_row + 1][position_col]
    perception_index = world_state[0] * 10 + world_state[1]
    perception_str += str(agent_perception[perception_index])

    return perception_str


@njit
def action(action_code, position_row, position_col, world, fitness_points, fitness_array, step_costs):
    """
    Simulates the action of an agent.
    :param action_code: the code for the action to take
    :param position_row: the row of the agent
    :param position_col: the col of the agent
    :param world: the world of the agent
    :param fitness_points: current fitness points of agent
    :param fitness_array: the array of the current fitness function
    :param step_costs: costs for the agent to take a single step

    :return: A list containing: the new row of the agent, the new columns of the agent, the new fitness points
    ---------------------------------------------------------------
    Actions code mapping:
    0: stay
    1: move left
    2: move right
    3: move up
    4: move down
    """
    # step left
    if action_code == 1:
        if position_col > 0:
            position_col -= 1
        fitness_points -= step_costs

    # step right
    if action_code == 2:
        if position_col < 9:
            position_col += 1
        fitness_points -= step_costs

    # step up
    if action_code == 3:
        if position_row > 0:
            position_row -= 1
        fitness_points -= step_costs

    # step down
    if action_code == 4:
        if position_row < 9:
            position_row += 1
        fitness_points -= step_costs

    x, y = world[position_row][position_col]
    fitness_points += fitness_array[x][y]

    return position_row, position_col, fitness_points


def estimate_fitness(agent_perception_genes, agent_decision_genes, explorations, actions,fitness_fcts, step_costs):
    """
    Estimates the fitness of a given agent.
    For each exploration, a world is created and the fitness is summed up over all actions.
    The average fitness for an exploration will be returned
    :param agent_perception_genes: agent's perception genome
    :param agent_decision_genes: agent's decision genome
    :param explorations: number of explorations
    :param fitness_fcts: list containig fitness function for each state
    :param actions: number of actions per exploration
    :param step_costs: what it costs an agent to take a singl step
    :return: agent's estimated fitness
    """

    total_fitness = 0
    for _ in range(explorations):
        world = World.create_world(RESOURCE_MIN, RESOURCE_MAX)
        position_row = 5
        position_col = 5
        current_agent_state = 0
        current_fitness_fct = fitness_fcts[current_agent_state]
        fitness_points = 0

        for action_counter in range(actions):
            # perceive
            perception_string = perceive(world, position_row, position_col, agent_perception_genes)

            action_index = current_agent_state * (NO_PERCEPTUAL_CATEGORIES ** 5) + int(perception_string,
                                                                                       NO_PERCEPTUAL_CATEGORIES)  # index at which to find the action in decision genome
            action_code = agent_decision_genes[action_index]  # the chosen action, represented as an int

            position_row, position_col, fitness_points = action(action_code, position_row, position_col, world,
                                                                fitness_points, current_fitness_fct, step_costs)

            # expand world if agent is at edge of the world
            if (position_col == 0 or position_col == world.shape[1] - 1
                    or position_row == 0 or position_row == world.shape[0] - 1):
                world = World.expand_world(world, RESOURCE_MIN, RESOURCE_MAX, WORLD_EXPANSION_SIZE)
                position_row += WORLD_EXPANSION_SIZE
                position_col += WORLD_EXPANSION_SIZE

            # change agent's state and fitness function every 10 actions
            if action_counter % 10 == 0:
                current_agent_state += 1
                current_agent_state = current_agent_state % len(fitness_fcts)
                current_fitness_fct = fitness_fcts[current_agent_state]

        total_fitness += fitness_points
    total_fitness /= explorations
    return total_fitness


def tournament(perceptions, decisions, fitness_scores, tournament_size, no_agents_per_generation, rng):
    """
    Gets the current perception and decision genes.
    Selects tournament_size many agents for the tournament.
    Selects and returns the fittest agent's perception and decision
    """
    tournament_agents_index = rng.choice(range(no_agents_per_generation), tournament_size, replace=False)
    tournament_perceptions = perceptions[tournament_agents_index]
    tournament_decisions = decisions[tournament_agents_index]

    tournament_agents_fitness = fitness_scores[tournament_agents_index]
    max_index = np.argmax(tournament_agents_fitness)

    return tournament_perceptions[max_index], tournament_decisions[max_index]


def crossover(genome_1, genome_2, rng):
    '''
    Creates 2 new crossover genes at a random point from 2 parent genes
    '''
    crossover_index = rng.integers(0, len(genome_1))

    genome_1_sub1 = genome_1[:crossover_index]
    genome_1_sub2 = genome_1[crossover_index:]
    genome_2_sub1 = genome_2[:crossover_index]
    genome_2_sub2 = genome_2[crossover_index:]

    offspring_1 = np.concatenate((genome_1_sub1, genome_2_sub2))
    offspring_2 = np.concatenate((genome_2_sub1, genome_1_sub2))

    return offspring_1, offspring_2


def mutate(genomes, mutation_indices, mutation_values):
    """
    Mutates the genomes of a whole generation in place.
    :param genomes: the genomes of a whole generation.
    must be a 2d numpy array (shape= (no_agents, PERCEPTION_GENOME_LENGTH)
    :param mutation_indices: the indices at which to mutate the genomes.
    must be a 2d numpy array filled with booleans (shape= (no_agents, PERCEPTION_GENOME_LENGTH)
    :param mutation_values: new values after the mutation.
    must be a 2d numpy array (shape= (no_agents, PERCEPTION_GENOME_LENGTH)
    """

    genomes[mutation_indices] = mutation_values[mutation_indices]


def simulate(no_agents_per_generation=DEFAULT_NO_AGENTS_PER_GENERATION,
             no_explorations_per_fitness_estimation=DEFAULT_NO_EXPLORATIONS_PER_FITNESS_ESTIMATION,
             no_actions_per_exploration=DEFAULT_NO_ACTIONS_PER_EXPLORATION,
             tournament_size=DEFAULT_TOURNAMENT_SIZE,
             mutation_probability=DEFAULT_MUTATION_PROBABILITY,
             no_generations=DEFAULT_NO_GENERATIONS,
             result_file_prefix=RESULT_FILE_PREFIX,
             fitness_fcts=DEFAULT_FITNESS_ARRAYS,
             step_costs=DEFAULT_STEP_COSTS,
             print_enabled=False):
    """
    Runs the simulation with the given parameters.
    The length of fitness_fcts is implicitly the number of states of agents
    :param no_agents_per_generation: number of agents per generation
    :param no_explorations_per_fitness_estimation: number of explorations per fitness estimation (each exploration is in a new world)
    :param no_actions_per_exploration: number of actions per exploration
    :param tournament_size: how many agents are drawn for the tournament
    :param mutation_probability: probability for each index of the genome to be mutated
    :param no_generations: number of generations for the simulation
    :param result_file_prefix: prefix for the result file. This will be followed by a timestamp and eventually additional information.
    :param fitness_fcts: The functions/arrays containing the fitness points for each state
    :param step_costs: how much it costs an agent to take a single step
    :param print_enabled: whether to print or not (set to False to reduce runtime)
    """

    MyUtils.PRINT_ENABLED = print_enabled

    my_print('++++++++++++++++++')
    my_print('Simulation starts')
    my_print('no_agents_per_generation: ' + str(no_agents_per_generation))
    my_print('no_explorations_per_fitness_estimation: ' + str(no_explorations_per_fitness_estimation))
    my_print('no_actions_per_exploration: ' + str(no_actions_per_exploration))
    my_print('tournament_size: ' + str(tournament_size))
    my_print('mutation_probability: ' + str(mutation_probability))
    my_print('no_generations: ' + str(no_generations))

    data_saver = None
    if SAVE_RESULTS:
        data_saver = DataSaver(result_file_prefix)
        data_saver.save('parameters', {
            'no_agents_per_generation': no_agents_per_generation,
            'no_explorations_per_fitness_estimation': no_explorations_per_fitness_estimation,
            'no_actions_per_exploration': no_actions_per_exploration,
            'tournament_size': tournament_size,
            'mutation_probability': mutation_probability,
            'no_generations': no_generations,
            'fitness_fcts': fitness_fcts.tolist()
        })

    # saving max and average fitness of each generation
    max_fitness_over_generation = []
    average_fitness_over_generation = []

    # get random generator with unique seed to use in all random processes
    seed = secrets.randbits(128)

    if SAVE_RESULTS:
        data_saver.save('seed', seed)
    rng = np.random.default_rng(seed)

    no_agent_states = len(fitness_fcts)
    decision_genome_length = no_agent_states * NO_PERCEPTUAL_CATEGORIES ** 5

    # already draw random indices and new values for mutation (for runtime optimization)
    perception_mutation_booleans = np.random.choice([False, True], size=(no_generations,
                                                                         no_agents_per_generation,
                                                                         PERCEPTION_GENOME_LENGTH),
                                                    p=[1 - mutation_probability, mutation_probability])

    perception_mutation_new_values = np.random.randint(0, NO_PERCEPTUAL_CATEGORIES, size=(no_generations,
                                                                                          no_agents_per_generation,
                                                                                          PERCEPTION_GENOME_LENGTH))

    decision_mutation_booleans = np.random.choice([False, True], size=(no_generations,
                                                                       no_agents_per_generation,
                                                                       decision_genome_length),
                                                  p=[1 - mutation_probability, mutation_probability])

    decision_mutation_new_values = np.random.randint(0, NO_ACTIONS, size=(no_generations,
                                                                          no_agents_per_generation,
                                                                          decision_genome_length))

    # create initial population
    fitness_scores = []
    agents_perception = rng.integers(0, NO_PERCEPTUAL_CATEGORIES,
                                     size=(no_agents_per_generation, PERCEPTION_GENOME_LENGTH))
    agents_decision = rng.integers(0, NO_ACTIONS, size=(no_agents_per_generation, decision_genome_length))

    for i in range(no_agents_per_generation):
        this_agents_fitness = estimate_fitness(agents_perception[i],
                                               agents_decision[i],
                                               no_explorations_per_fitness_estimation,
                                               no_actions_per_exploration,
                                               step_costs)
        fitness_scores.append(this_agents_fitness)

    agents_perception = np.array(agents_perception)
    agents_decision = np.array(agents_decision)

    # simulate generations
    for generation in range(no_generations):

        my_print('generations ' + str(generation))
        max_fitness_over_generation.append(np.max(fitness_scores))
        average_fitness_over_generation.append(np.average(fitness_scores))
        my_print('max: ' + str(np.max(fitness_scores)))
        my_print('avg: ' + str(np.average(fitness_scores)))

        if SAVE_RESULTS:
            data_saver.save('perception_gener' + str(generation), agents_perception)
            data_saver.save('decision_gener' + str(generation), agents_decision)
            data_saver.save('fitness_gener' + str(generation), fitness_scores)

        new_generation_perception = []
        new_generation_decision = []
        for i in range(int(no_agents_per_generation / 2)):
            # select 2 parents via tournament selection
            parent_1_perception, parent_1_decision = tournament(agents_perception, agents_decision,
                                                                np.array(fitness_scores), tournament_size,
                                                                no_agents_per_generation, rng)
            parent_2_perception, parent_2_decision = tournament(agents_perception, agents_decision,
                                                                np.array(fitness_scores), tournament_size,
                                                                no_agents_per_generation, rng)

            # crossover
            offspring_1_perception, offspring_2_perception = crossover(parent_1_perception, parent_2_perception, rng)
            offspring_1_decision, offspring_2_decision = crossover(parent_1_decision, parent_2_decision, rng)

            new_generation_perception.append(offspring_1_perception)
            new_generation_decision.append(offspring_1_decision)

            new_generation_perception.append(offspring_2_perception)
            new_generation_decision.append(offspring_2_decision)

        agents_perception = np.array(new_generation_perception)
        agents_decision = np.array(new_generation_decision)

        # mutate
        mutate(agents_perception, perception_mutation_booleans[generation],
               perception_mutation_new_values[generation])
        mutate(agents_decision, decision_mutation_booleans[generation], decision_mutation_new_values[generation])

        fitness_scores = []
        for i in range(no_agents_per_generation):
            fitness_scores.append(
                estimate_fitness(agents_perception[i],
                                 agents_decision[i],
                                 no_explorations_per_fitness_estimation,
                                 no_actions_per_exploration,
                                 fitness_fcts,
                                 step_costs
                                 ))

    print('Computation-time: ' + str(datetime.datetime.now() - start_time))

    # save fitness data and dump data to file
    if SAVE_RESULTS:
        data_saver.save('max_fitness_over_generation', max_fitness_over_generation)
        data_saver.save('avg_fitness_over_generation', average_fitness_over_generation)

        data_saver.dump_data()
