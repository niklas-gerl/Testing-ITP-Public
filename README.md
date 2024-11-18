# Genetic Algorithm to test Interface Theory of Perception

## Overview
This project contains the code to run genetic algorithms to simulate how the environment, namely the fitness function,
influences the evolutionary development of perception. It is possible to produce evidence for the Interface Theory of Perception (ITP), 
which is proposed by Donald Hoffman. 

## Project Objectives
The key question is whether the agents of an environment will develop veridical perception or strict interface perceptions.

## Requirements
This program requires an installed python3 version (python 3.12 recommended). Additionally, it requires the packages
[numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), [numba](https://numba.pydata.org/), [scipy](https://scipy.org/).
Use `pip install numpy matplotlib numba scipy` to install the required packages.


## The Genetic Algorithm 

### World
The world is an infinitely large grid space. Each grid contains a bounded amount of resources x and y. An Agent perceives the 4
cells around it and the cell of the current position. It can either move to one of these cells or stay. Depending on the resource quantities of x and y 
of the cell, the agent receives fitness points. 

### Agents
All agents of a simulation have the same shape. This means that the following parameter are set once in the 
simulation for all agents: 
* `NO_PERCEPTUAL_CATEGORIES`: How many different perceptual categories an agent can have for a cell. 
Intuitively think of these as colors. If agents have n perceptual categories, they have n colors in which they can see a cell. 

* `no_agent_states`: How many possible states each agent has. An agent is always in exactly 1 state. The active state defines which fitness function is currently used for the agent. State has no other effects. It is switched every 10 actoins.
* `fitness_fcts`: Multiple fitness arrays (one for each agent state) are stored in a list. A fitness array defines how many fitness points an agent receives from which resource quantities. 
* `agents_peception`: An array containing all perception genomes for the current generation. A perception genome is an array of length n_x * n_y, where n_x (n_y) is the number of possible values for x (y). Thus, it maps each possible (x,y) to a perceptual category.  
* `agents_decision`: An array containing all decision genomes for the current generation. A decision genome is an array of length (`NO_AGENT_STATES` * ` NO_PERCEPTUAL_CATEGORIES ` )^ 5. It maps each possible perceptional state to an action. The perceptional is defined by the 5 perceptual categories of the surrounding and current cell. 

### Evolutionary Process
The variable `no_generations` determines how often the following process is repeated. 
#### Evaluate fitness of all agents
To estimate the fitness of an agent, `no_explorations_per_fitness_estimation`-many worlds are created and in each world the agent does `no_actions_per_exploration`-many actions. 
After each action, the fitness points of its current cell are evaluated and counted. Additionally, each step the agent takes
costs `step_costs`-many fitness points. 

#### Use tournament selection to choose parents for offspring for next generation
For a tournament, `tournament_size`-many agents get selected with equal probability. The agent with the highest fitness estimate wins the tournament and is now a parent. An agent can be a parent multiple times,
which is likely if it has a high fitness estimate. 

#### Create new generation by crossover
All parents become a partner and all pairs will create 2 offsprings. The 2 offsprings will be created by splitting the perception genome of both parents,
and recombining them such that an offspring's perception genome consists partly of the first parent's and partly of the second parent's perception genome.The same is done for the decision genome.  


#### Mutate new generation
All agents of the new population undergo random mutation. This means that each index of their genome can be changed to a new random value by a certain (usually low) probability, which is set in `mutation_probability`.

## How to Run the Genetic Algorithm

### Method `simulate(...)`
With `simulate(...)` you can directly start a single simulation. You cnn call it without any parameters, which will lead
to the simulation using the default parameters which are set in the beginning of the file.
You can also adjust certain parameters of the simulation by using the parameters of the method simulation.

### Method `repeat_simulations(...)`
Use this method if you want to use the current Default values of the simulation but repeat it multiple times.
This method should be preferred over calling `simulation(...)` multiple times because it uses parallelization and therefor
reduces runtime enormously. 

### Method `multiple_simulations(...)`
With this method you can run multiple simulations each with a different parameter set. 
This method also uses parallelization. 



## Data Storage

The output data is generally a dictionary which is stored as a JSON file. 
The standard path for the result file is `data/records/<result_file_prefix><timestamp>.txt`
During the simulation, data storage is handled by an instance of `MyUtils.DataSaver`

This is the scheme for the result dictionary/JSON file: 
- `parameters`: *(dict/object)* - Parameters used in the simulation.
  - `number_of_agents_per_generation`: (int)
  - `number_of_explorations_per_fitness_estimation`: *(int)*
  - `number_of_actions_per_exploration`: *(int)*
  - `tournament_size`*: (int)*
  - `mutation_probability`*: (int)*
  - `number_of_generations`*: (int)*
  - `fitness_fct`*: (list)
- `seed`: *(int)* - The seed used for random processes
- `perception_gener0`: *(list)* - Contains perception genome of each agent of generation 0
- `decision_gener0`: *(list)* - Contains decision genome of each agent of generation 0
- `fitness_gener0`: *(list)* - Contains fitness estimation of each agent of generation 0
- ...
- `perception_gener500`: *(list)* - Contains perception genome of each agent of generation 500
- `decision_gener500`: *(list)* - Contains decision genome of each agent of generation 500
- `fitness_gener500`: *(list)* - Contains fitness estimation of each agent  of generation 500
- `max_fitness_over_generation`: *(list)* - Contains the maximum fitness of population for each generation
- `avg_fitness_over_generation`: *(list)* - Contains the average fitness of population for each generation

Note that for every generation i, there are 3 key value pairs in the dict: 
- `perception_generi`: *(list)* 
- `decision_generi`: *(list)* 
- `fitness_generi`: *(list)* 

The perception, decision and fitness of an agent are all stored at the same index in all 3 lists. 


## Analysis
In the Analysis file, there are a variety of methods to analyse the data and to create plots.
The `PERCEPTION_CMAP` defines which perceptual categories are presented as which colors in the plots.
To use any of the analysis function, import a result file and use the `json` module to import the data into a dict called `data`.
This dict will be automatically used for analysis and plotting. 
For an example usage, see the Analysis main method.
























