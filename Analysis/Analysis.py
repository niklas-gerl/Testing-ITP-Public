"""
Methods to analyse the results of a simulation.
The methods access a variable called 'data', which must contain the experiment results as a dictionary.
"""

import mpl_toolkits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import FitnessFunctions
from GeneticAlgorithm import estimate_fitness
import CriticalRealists

# perception color bar
start_color = (0 / 255, 125 / 255, 0 / 255, 1)  # Green: RGBA (0, 125, 0, 255)
middle_color = (1, 1, 1, 1)  # White: RGBA (255, 255, 255, 255)
end_color = (171 / 255, 0 / 255, 0 / 255, 1)  # Red: RGBA (171, 0, 0, 255)
PERCEPTION_CMAP = LinearSegmentedColormap.from_list('custom_cmap', [start_color, middle_color, end_color])


def color_function(perception):
    """
    Is used by the plot single perception methods
    """
    res = np.empty(shape=(10, 10))
    for x in range(0, 10):
        for y in range(0, 10):
            res[x][y] = perception[x * 10 + y]
    return res


def add_headers(
        fig,
        *,
        row_headers=None,
        col_headers=None,
        row_pad=1,
        col_pad=5,
        rotate_row_headers=True,
        **text_kwargs
):
    """
    adds headers to figure rows and columns
    Based on https://stackoverflow.com/a/25814386
    """

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row() and (
                isinstance(ax, mpl_toolkits.mplot3d.Axes3D or ax.get_title == 'Average Perception')):
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def plot_fitness_fct(fitness_fct, title: str = '', show=True, axes=None, fig = None):
    """
    Creates a 3d bar plot and 2d colormap plot for a fitness function.
    """
    if axes == None:
        fig, axes = plt.subplots(2)
        axes[1] = fig.add_subplot(2,1, 2, projection='3d')

    x = np.arange(fitness_fct.shape[1])
    y = np.arange(fitness_fct.shape[0])
    X, Y = np.meshgrid(x, y)

    #axes[0].contourf(X, Y, fitness_fct, cmap='viridis')
    #axes[0].colorbar()  # Add a colorbar for reference

    x = np.arange(fitness_fct.shape[1])
    y = np.arange(fitness_fct.shape[0])
    X, Y = np.meshgrid(x, y)

    # Flatten the data and coordinates
    x = X.flatten()
    y = Y.flatten()
    z = np.zeros_like(x)
    dx = dy = 0.75  # Width and depth of bars

    # 3D plot
    # Create a grid for x and y coordinates
    x = np.arange(fitness_fct.shape[1])
    y = np.arange(fitness_fct.shape[0])
    X, Y = np.meshgrid(x, y)

    # Adjust the position of bars
    x_adjust = 0.5
    y_adjust = 0.5

    # Plot the 3D bar plot
    pcm = axes[1].bar3d(X.flatten() - x_adjust, Y.flatten() - y_adjust, np.zeros_like(fitness_fct.flatten()), 1, 1,
                  fitness_fct.flatten(),
                  color=plt.get_cmap('viridis')(fitness_fct/100).reshape(100,4), shade=True)

    # Set labels and title
    axes[1].set_xlabel('x-property')
    axes[1].set_ylabel('y-property')
    axes[1].set_zlabel('fitness')
    axes[1].set_title(title)



    if show and fig is not None:
        fig.show()

    return pcm


def plot_fitness(show_fig=True, save=""):
    """
    plot the max and avg fitness development of a simulation over generations
    """
    max_fitness = []
    avg_fitness = []

    for key, value in data.items():
        if key.startswith('fitness'):
            max_fitness.append(max(value))
            avg_fitness.append(sum(value) / len(value))

    generations = range(len(avg_fitness))

    # Plotting both graphs on the same plot
    plt.plot(generations, avg_fitness, label='average fitness')
    plt.plot(generations, max_fitness, label='max fitness')

    # Adding labels and legend
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.legend()

    if save != '':
        plt.savefig('plots/' + str(save) + '.png', format='png')

    if show_fig:
        plt.show()


def create_perception_axes(axis_size: int, title: str, perception, show_fig=True, save=""):
    """
    creates the color map for a single perception genome
    Shows the figure if show_fig is True.
    Saves the figure in directory /plots if save is not ""

    :param axis_size: size of x and y axis
    :param perception: the perception genome to plot
    :param show_fig: if the figure should be shown
    :param save: if and under which name the figure should be saved
    """
    fig, ax = plt.subplots()

    # Create a grid of x and y values
    x = np.linspace(0, axis_size, axis_size + 1)
    y = np.linspace(0, axis_size, axis_size + 1)

    X, Y = np.meshgrid(x, y)

    colors = color_function(perception)

    pcm = ax.pcolormesh(X, Y, colors)

    fig.colorbar(pcm, ax=ax)

    ax.set_title(title)

    ax.plot()

    if save != "":
        fig.savefig('plots/' + str(save) + '.png', format='png')

    if show_fig:
        fig.show()

    plt.close(fig)

    return fig


def create_perception_axes_inplace(ax, axis_size: int, title: str, perception, cmap=PERCEPTION_CMAP):
    """
    creates the color map for a single perception
    Shows the figure if show_fig is True.
    Saves the figure in directory plots if save is not ""

    :param axis_size: size of x and y axis
    :param perception: the perception to plot
    :param show_fig: if the figure should be shown
    :param save: if and under which name the figure should be saved
    """

    # Create a grid of x and y values
    x = np.linspace(0, axis_size, axis_size + 1)
    y = np.linspace(0, axis_size, axis_size + 1)

    X, Y = np.meshgrid(x, y)

    colors = color_function(perception)

    linewidths = 0  # use 0.1 to show grid on the plot
    pcm = ax.pcolormesh(X, Y, colors, edgecolors='k', linewidths=linewidths, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel('x-Property')
    ax.set_ylabel('y-Property')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # display integer values on axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # display integer values on axis

    # set ticks on axis at center of fields
    ax.set_xticks(np.arange(10) + 0.5)
    ax.set_xticklabels(np.arange(10))

    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_yticklabels(np.arange(10))

    ax.plot()

    return pcm


def plot_perception_of_generation(generation: int):
    """
    creates the color map for all perceptions for a generation and saves them in a dir
    """
    for i in range(200):
        print(i)
        perception = data['perception_gener' + str(generation)][i]
        create_perception_axes(10, 'agent ' + str(i), perception, show_fig=False, save="agent" + str(i))


def average_perception_over_generation(generation: int):
    """
    Averages the perception of a generation
    """
    return np.average(data['perception_gener' + str(generation)], axis=0)


def get_fittest_agent_of_generation(generation: int):
    """
    Returns the perception, decision and fitness of fittest agent of a generation
    """

    fitness_scores = data['fitness_gener' + str(generation)]
    max_fitness = np.max(fitness_scores)
    max_index = fitness_scores.index(max_fitness)

    return data['perception_gener' + str(generation)][max_index], data['decision_gener' + str(generation)][
        max_index], max_fitness


def print_decision_table(decisions):
    """
    prints the decision of an agent in a human-readable format and creates a fig with the decision table
    """

    fig, ax = plt.subplots(figsize=(6, 8))

    genes_action_mapping = {
        0: 'stay',
        1: 'move left',
        2: 'move right',
        3: 'move up',
        4: 'move down',
    }

    genes_color_mapping = {
        '0': start_color,
        '1': end_color,
        'd0': start_color,
        'd1': end_color,

    }

    table = []
    cellColours = []
    for i in range(32):
        decimal_rep = np.base_repr(i, 2)
        decimal_rep = str(decimal_rep)
        while len(decimal_rep) < 5:
            decimal_rep = '0' + decimal_rep

        print_str = 'current:' + str(decimal_rep[0]) + '  left:' + str(decimal_rep[1]) + '  right:' + str(
            decimal_rep[2]) + '  up:' + str(decimal_rep[3]) + '  down:' + str(decimal_rep[4]) + '-->' + \
                    genes_action_mapping[decisions[i]]
        print(print_str)

        table.append([decimal_rep[0], decimal_rep[1], decimal_rep[2], decimal_rep[3], decimal_rep[4],
                      genes_action_mapping[decisions[i]]])
        cellColours.append([genes_color_mapping[decimal_rep[0]],
                            genes_color_mapping[decimal_rep[1]],
                            genes_color_mapping[decimal_rep[2]],
                            genes_color_mapping[decimal_rep[3]],
                            genes_color_mapping[decimal_rep[4]],
                            genes_color_mapping['d' + decimal_rep[decisions[i]]]])

    ax.axis('off')
    colLabels = ['current', 'left', 'right', 'up', 'down', 'decision']

    colors = []
    for i in range(16):
        colors.append(['w'] * 6)
        colors.append(['lightgrey'] * 6)

    table = ax.table(cellText=table, colLabels=colLabels, cellColours=cellColours, loc='center',
                     colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])

    for cell in table._cells.values():
        cell.set_text_props(ha='center', va='center')
    fig.show()


def analyse_decisions(decisions, no_perceptual_categories: int):
    """
    analyses which perceptual state is preferred by decision genome.
    :param decisions: the decision genome to analyse
    :no_perceptual_categories: number of perceptual categories
    :return: a list containing at index i the number of perceptual states for which perceptual category i is chosen
    """
    seeking = [0] * no_perceptual_categories

    for i in range(len(decisions)):
        decimal_rep = np.base_repr(i, 2)
        decimal_rep = str(decimal_rep)
        while len(decimal_rep) < 5:
            decimal_rep = '0' + decimal_rep

        decision = decimal_rep[decisions[i]]
        seeking[int(decision)] += 1

    return seeking


def plot_avg_perception_over_generations(path):
    """
    plots the average perception of all generations and saves them in a dir
    """
    number_of_generations = data['parameters']['no_generations']

    for generation in range(number_of_generations):
        print(generation)
        avg = average_perception_over_generation(generation)
        create_perception_axes(10, 'perception_gener' + str(generation), avg, False, path + 'avg' + str(generation))


def plot_perception_over_generations_together(generations=[0, 1, 25, 50, 75, 100, 200, 300, 499]):
    """
    Creates a fig which contains average and fittest Perception over certain generations
    :param generations: the generations to plot
    """

    ncols = len(generations)
    fig, axes = plt.subplots(2, ncols, figsize=(28, 6))
    fig.subplots_adjust(hspace=0.5, left=0.02, right=0.98)
    print(list(generations))
    for generation, i in zip(generations, range(len(generations))):
        perception_avg = average_perception_over_generation(generation)
        perception_max, _, _ = get_fittest_agent_of_generation(generation)

        avg_axes = axes[0][i]
        max_axes = axes[1][i]

        # force axes to be squared
        avg_axes.set_aspect('equal')
        max_axes.set_aspect('equal')

        colorbar_avg = create_perception_axes_inplace(avg_axes, 10, '', perception_avg)

        create_perception_axes_inplace(max_axes, 10, '', perception_max)

    col_headers = list(map(lambda g: 'Generation ' + str(g), list(generations)))
    add_headers(fig, row_headers=['Average Perception', 'Fittest Perception', ''], col_headers=col_headers, col_pad=20,
                row_pad=18, size=15)

    fig.colorbar(colorbar_avg, ax=[axes[0, ncols - 1], axes[1, ncols - 1]], pad=0.15)

    fig.show()


def plot_fittest_perception_over_generations(path):
    """
    plots the fittest perception of all generations
    """
    number_of_generations = data['parameters']['no_generations']

    for generation in range(number_of_generations):
        print(generation)
        max_perception, _, _ = get_fittest_agent_of_generation(generation)
        create_perception_axes(10, 'perception_gener' + str(generation), max_perception, False,
                               path + 'max' + str(generation))


def plot_realist_perceptions():
    """
    Plots some realist perceptions and their fitness estimation
    """

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.0, top=0.97, bottom=0.03, left=0.02, right=0.98)

    realists = []
    realists.append(CriticalRealists.get_critical_realist_x(3))
    realists.append(CriticalRealists.get_critical_realist_x(5))
    realists.append(CriticalRealists.get_critical_realist_x(8))
    realists.append(CriticalRealists.get_critical_realist_y(3))
    realists.append(CriticalRealists.get_critical_realist_y(5))
    realists.append(CriticalRealists.get_critical_realist_y(8))
    realists.append(CriticalRealists.get_critical_realist_xy(5))
    realists.append(CriticalRealists.get_critical_realist_xy(10))
    realists.append(CriticalRealists.get_critical_realist_xy(15))

    decision_pref_1 = CriticalRealists.decision_prefer_x('1')
    decision_pref_0 = CriticalRealists.decision_prefer_x('0')

    max_fitness = 5800  # maximum fitness by other agents

    for realist, ax in zip(realists, axes.flatten()):
        fitness_with_pref_0 = estimate_fitness(realist, decision_pref_0, 10000, 100, FitnessFunctions.normal_1_arr)
        fitness_with_pref_1 = estimate_fitness(realist, decision_pref_1, 10000, 100, FitnessFunctions.normal_1_arr)
        create_perception_axes_inplace(ax, 10, 'Fitness: ' + str(
            np.round(max(fitness_with_pref_0, fitness_with_pref_1) / max_fitness * 100, 0)) + '%', realist)

    for ax in axes.flatten():
        ax.set_aspect('equal')
    fig.show()


def analyse_various_FitnessFunctions():
    """
    creates the plots for analysing the 11 experiments with varying fitness functions
    """
    global data

    simulation_data_dir = 'data/records/pub2/'

    for filename, i in zip(os.listdir(simulation_data_dir), 'abkcdefghij'):
        with open(simulation_data_dir + filename) as json_file:
            data = json.load(json_file)

            fitness_fct = np.array(data['parameters']['fitness_fct'])
            fig, axes = plt.subplots(2, 2, figsize=(11, 10),
                                     facecolor=(0.8941176470588236, 0.8941176470588236, 0.8941176470588236, 1.0))
            fig.subplots_adjust(hspace=0.35)

            fig.suptitle('Simulation (' + i + ')', fontsize=16)

            plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.9, wspace=0.0, hspace=0.25)

            axes[0, 0].axis('off')
            axes[0, 0] = fig.add_subplot(2, 2, 1, projection='3d',
                                         facecolor=(0.8941176470588236, 0.8941176470588236, 0.8941176470588236, 1.0))
            create_perception_axes_inplace(axes[0, 1], 10, '', fitness_fct.flatten(), 'viridis')

            pcm = plot_fitness_fct(fitness_fct, '', False, [axes[0, 1], axes[0, 0]], fig)
            fitness_colorbar = fig.colorbar(pcm, ax=axes[0, 1], pad=0.15)
            fitness_colorbar.set_ticklabels(np.linspace(0, 100, 6))
            fitness_colorbar.set_label('Fitness Payoff')

            fittest_perception, fittest_decision, _ = get_fittest_agent_of_generation(499)
            avg_perception = average_perception_over_generation(499)

            print(analyse_decisions(fittest_decision, 2))

            for ax in axes.flatten()[1:4]:
                ax.set_aspect('equal')
                ax.margins(0.1)

            pcm = create_perception_axes_inplace(axes[1, 0], 10, 'Average Perception Gen. 500', fittest_perception,
                                                 cmap=PERCEPTION_CMAP)
            create_perception_axes_inplace(axes[1, 1], 10, 'Fittest Perception Gen. 500', avg_perception,
                                           cmap=PERCEPTION_CMAP)
            fig.colorbar(pcm, ax=axes[1, 1], pad=0.15).set_label('Perceptual Category')

            fig.show()
            fig.savefig('plots/pub2/sim_' + str(i))


def ITvsREAL():
    """recreates the interface and realist plot (adapted by Hoffman(2015)) on a 1-d normal fitness function."""
    # Define the data
    x = np.linspace(0, 100, 500)
    y = 100 * np.exp(-((x - 50) ** 2) / (2 * (20) ** 2))

    colors_realist = ['red', 'orange', 'green', 'blue']
    borders_realist = [0, 25, 50, 75, 100]

    colors_interface = ['red', 'orange', 'green', 'blue', 'green', 'orange', 'red']
    borders_interface = [0, 12.5, 25, 37.5, 62.5, 75, 87.5, 100]

    titles = ['Critical Realist', 'Strict Interface']

    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    for ax, colors, borders, title in zip(axs, [colors_realist, colors_interface], [borders_realist, borders_interface],
                                          titles):
        for i in range(len(colors)):
            ax.axvspan(borders[i], borders[i + 1], color=colors[i], alpha=0.5)
        ax.plot(x, y, color='black')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)

        ax.set_title(title)

        # Set x-ticks on the perceptual borders
        ax.set_xticks(borders)

        # Set 10 ticks on the y-axis
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ])

        # Set labels
        ax.set_xlabel('Resource Quantity')
        ax.set_ylabel('Payoff')

    plt.tight_layout()
    plt.show()


def plot_multiple_fitness_fcts():
    """
    plots all fitness functions, the average fitness function and the avg and fittest perception of last generation of a simulation with dynamic fitness functions and also
    """
    fitness_arrays = data['parameters']['fitness_fcts']
    fitness_arrays.append(FitnessFunctions.get_average_fitness_fct_normalized(fitness_arrays))
    fitness_arrays = np.array(fitness_arrays)
    ncols = len(fitness_arrays) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 6, 10), layout='constrained')
    fig.subplots_adjust(left=0.1, right=0.99)

    # plot fitness functions
    for i in range(ncols - 1):
        axes[0, i].axis('off')
        axes[0, i] = fig.add_subplot(2, ncols, i + 1, projection='3d')
        pcm_fitness = create_perception_axes_inplace(axes[1, i], 10, '', fitness_arrays[i].flatten(), 'viridis')
        plot_fitness_fct(fitness_arrays[i], '', False, [axes[1, i], axes[0, i]], fig)

        axes[1, i].set_aspect('equal')

    # plot perception
    avg_perception = average_perception_over_generation(499)
    fittest_perc, _, _ = get_fittest_agent_of_generation(499)

    pcm_perception = create_perception_axes_inplace(axes[0, ncols - 1], 10, 'Average Perception', avg_perception,
                                                    PERCEPTION_CMAP)
    create_perception_axes_inplace(axes[1, ncols - 1], 10, 'Fittest Perception', fittest_perc, PERCEPTION_CMAP)

    axes[0, ncols - 1].set_aspect('equal')
    axes[1, ncols - 1].set_aspect('equal')

    col_headers = list(map(lambda i: 'Fitness Function in State ' + str(i + 1), list(range(ncols))))
    col_headers[ncols - 2] = 'Averaged Fitness Function'
    col_headers[ncols - 1] = 'Resulting Perceptions'

    add_headers(fig, col_headers=col_headers, size=15)

    pcm_ax = fig.colorbar(pcm_fitness, ax=axes[:, :-1])
    fig.colorbar(pcm_perception, ax=axes[:, -1:], pad=0.15)

    fig.show()

    return fig


def analyse_state_dependend_decision(decisions, no_states):
    # chunk decision genome into genomes for each state
    state_decisions = []
    chunk_size = len(decisions) / no_states
    for i in range(no_states):
        start_index = int(chunk_size * i)
        end_index = int((chunk_size * (i + 1)))
        state_decisions.append(decisions[start_index:end_index])

    # analyse each state decision
    for state in range(no_states):
        preferences = analyse_decisions(state_decisions[state], 2)
        if preferences[0] < preferences[1]:
            pref_color = 'red'
        else:
            pref_color = 'green'
        print('state ' + str(state) + str(preferences) + ' -> ' + pref_color)


if __name__ == '__main__':
    for filename in os.listdir('C:/Users/nikla/OneDrive/Uni/Module/Bachelorarbeit/data/experiment_set_2/NOF10'):
        with open(
                'C:/Users/nikla/OneDrive/Uni/Module/Bachelorarbeit/data/experiment_set_2/NOF10/' + filename) as json_file:
            data = json.load(json_file)
            # do analysis and plotting here
