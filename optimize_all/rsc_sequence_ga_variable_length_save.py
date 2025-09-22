
from deap import base, creator, tools, algorithms
import numpy as np
import random
import RSC_sim

# Define allowed pulse types
allowed_pulses = [(0, -6), (0, -5), (0, -4), (0, -3), (0, -2), (0, -1),
                  (1, -6), (1, -5), (1, -4), (1, -3), (1, -2), (1, -1),
                  (2, -9), (2, -8), (2, -7), (2, -6), (2, -5), (2, -4), (2, -3), (2, -2), (2, -1)]

# Build original sequence

# Build the original sequence manually
import RSC_functions
import sequence_functions




sequence = sequence_functions.load_sequence('sequences/best_sequence_1.txt')

initial_indices = [allowed_pulses.index(p) for p in sequence]

# Allow genome lengths to vary
MIN_PULSES = 50
MAX_PULSES = 400
mol_num = 1000
ngen = 50
n_pop = 20
def evaluate_sequence(individual):
    pulses = [allowed_pulses[i] for i in individual]
    sequence = [
        [axis, delta_n, RSC_sim.pulse_time(axis, delta_n)]
        for (axis, delta_n) in pulses
    ]
    mols = RSC_sim.initialize_thermal([25e-6, 25e-6, 25e-6], mol_num)
    _, _, ground_counts, _ = RSC_sim.apply_raman_sequence(mols, sequence)
    return (ground_counts[-1],)

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene", lambda: random.randint(0, len(allowed_pulses) - 1))
toolbox.register("individual", lambda: creator.Individual(
    [random.randint(0, len(allowed_pulses) - 1)
     for _ in range(random.randint(MIN_PULSES, MAX_PULSES))]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_sequence)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(allowed_pulses) - 1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
def run_ga():
    pop = toolbox.population(n=n_pop)
    seed_individual = creator.Individual(initial_indices[:])
    pop[0] = seed_individual  # Use known-good as seed

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=ngen,
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    best_sequence = [
        [axis, delta_n, RSC_sim.pulse_time(axis, delta_n)]
        for (axis, delta_n) in [allowed_pulses[i] for i in best_ind]
    ]
    
    # Save best sequence as [[axis, delta_n], ...]
    best_logical = [[axis, delta_n] for (axis, delta_n) in [allowed_pulses[i] for i in best_ind]]
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"sequences/sequence_var_length_{timestamp}.txt"
    with open(filename, "w") as f:
        for item in best_logical:
            f.write(str(item) + "\n")
    return best_sequence

if __name__ == "__main__":
    best_seq = run_ga()
    print("Best Sequence Found:")
    for pulse in best_seq:
        print(pulse)
