import GeneticAlgorithm
import GeneticAlgorithmWrapper

if __name__ == '__main__':
    GeneticAlgorithm.simulate(
        no_agents_per_generation=10,
        no_explorations_per_fitness_estimation=10,
        no_actions_per_exploration=10,
        tournament_size=3,
        no_generations=3
    )
