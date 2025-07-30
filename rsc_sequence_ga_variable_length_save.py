
from deap import base, creator, tools, algorithms
import numpy as np
import random
from RSC_functions import initialize_thermal, apply_raman_sequence, pulse_time

# Define allowed pulse types
allowed_pulses = [(0, -3), (0, -2), (0, -1),
                  (1, -3), (1, -2), (1, -1),
                  (2, -5), (2, -4), (2, -3), (2, -2), (2, -1)]

# Build original sequence

# Build the original sequence manually
import RSC_functions

sequence = []
# 10 * XY
for _ in range(10):
    sequence.append((0, -3))
    sequence.append((1, -3))
    sequence.append((0, -2))
    sequence.append((1, -2))

# 5 * XYZ1
for _ in range(5):
    sequence.append((2, -5))
    sequence.append((0, -2))
    sequence.append((2, -4))
    sequence.append((1, -2))
    sequence.append((2, -5))
    sequence.append((0, -1))
    sequence.append((2, -4))
    sequence.append((1, -1))

# 5 * XYZ2
for _ in range(5):
    sequence.append((2, -4))
    sequence.append((0, -2))
    sequence.append((2, -3))
    sequence.append((1, -2))
    sequence.append((2, -4))
    sequence.append((0, -1))
    sequence.append((2, -3))
    sequence.append((1, -1))

# 10 * XYZ3
for _ in range(10):
    sequence.append((2, -3))
    sequence.append((0, -2))
    sequence.append((2, -2))
    sequence.append((1, -2))
    sequence.append((2, -3))
    sequence.append((0, -1))
    sequence.append((2, -2))
    sequence.append((1, -1))

# 10 * XYZ4
for _ in range(10):
    sequence.append((2, -2))
    sequence.append((0, -2))
    sequence.append((2, -1))
    sequence.append((1, -2))
    sequence.append((2, -2))
    sequence.append((0, -1))
    sequence.append((2, -1))
    sequence.append((1, -1))

initial_indices = [allowed_pulses.index(p) for p in sequence]

# Allow genome lengths to vary
MIN_PULSES = 50
MAX_PULSES = 300

def evaluate_sequence(individual):
    pulses = [allowed_pulses[i] for i in individual]
    sequence = [
        [axis, delta_n, pulse_time(axis, delta_n)]
        for (axis, delta_n) in pulses
    ]
    mols = initialize_thermal([25e-6, 25e-6, 25e-6], 100)
    _, _, ground_counts = apply_raman_sequence(mols, sequence)
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
    pop = toolbox.population(n=20)
    seed_individual = creator.Individual(initial_indices[:])
    pop[0] = seed_individual  # Use known-good as seed

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=30,
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    best_sequence = [
        [axis, delta_n, pulse_time(axis, delta_n)]
        for (axis, delta_n) in [allowed_pulses[i] for i in best_ind]
    ]
    
    # Save best sequence as [[axis, delta_n], ...]
    best_logical = [[axis, delta_n] for (axis, delta_n) in [allowed_pulses[i] for i in best_ind]]
    with open("best_sequence_logical.txt", "w") as f:
        for item in best_logical:
            f.write(str(item) + "\n")
    return best_sequence

if __name__ == "__main__":
    best_seq = run_ga()
    print("Best Sequence Found:")
    for pulse in best_seq:
        print(pulse)
