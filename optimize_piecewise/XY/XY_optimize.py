from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable

import random
import numpy as np
from datetime import datetime
import json
from deap import base, creator, tools, algorithms
import os
import RSC_sim

# -----------------------------
# Problem definition / fitness
# -----------------------------

# Allowed pulses
ALLOWED_PULSES: List[Tuple[int, int]] = [
    (0, -5), (0, -4), (0, -3), (0, -2),
    (1, -5), (1, -4), (1, -3), (1, -2),
]

@dataclass
class GAConfig:
    mol_num: int = 1000
    temps: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    ngen: int = 10
    mu: int = 40         # parent pop size
    lambda_: int = 20   # offspring per gen
    cxpb: float = 0.65
    mutpb: float = 0.30
    tournament_k: int = 3
    n_jobs: int = 8      # 0 = serial; >0 uses multiprocessing
    # Fitness shaping
    len_penalty: float = 0.0   # penalize long sequences (lambda for L0 term)
    time_penalty: float = 0.0  # penalize total time (per second)
    # Adaptive mutation schedule
    mutpb_decay: float = 0.98  # per generation
    mut_indpb: float = 0.1     # per-gene mutation probability
    # Early stopping
    patience: int = 10


def cost_function(mol_list: Iterable) -> int:
    """Your original success count: n_x<=1 and n_y<=1.
    Modify if you want z or harsher thresholds.
    """
    good = 0
    for mol in mol_list:
        if mol.n[0] <= 1 and mol.n[1] <= 1:
            good += 1
    return good


def sequence_from_indices(indices: List[int]) -> List[Tuple[int, int, float]]:
    pulses = [ALLOWED_PULSES[i] for i in indices]
    return [[axis, dn, RSC_sim.pulse_time(axis, dn)] for axis, dn in pulses]


# Simple cache to avoid recomputing identical individuals
_eval_cache: Dict[Tuple[int, ...], float] = {}


def evaluate_indices(indices: List[int], cfg: GAConfig) -> float:
    key = tuple(indices)
    if key in _eval_cache:
        return _eval_cache[key]

    seq = sequence_from_indices(indices)
    mols = RSC_sim.initialize_thermal(list(cfg.temps), cfg.mol_num)
    # apply_raman_sequence returns multiple things; we only need terminal mols
    _ = RSC_sim.apply_raman_sequence(mols, seq)

    score = float(cost_function(mols))

    # Optional penalties (encourage shorter/faster sequences)
    if cfg.len_penalty:
        score -= cfg.len_penalty * len(indices)
    if cfg.time_penalty:
        total_t = sum(s[2] for s in seq)
        score -= cfg.time_penalty * total_t

    _eval_cache[key] = score
    return score


# -----------------------------------
# GA with improved meta‑parameters
# -----------------------------------

def build_toolbox(N_PULSES: int, cfg: GAConfig, seed_indices: List[int] | None = None) -> base.Toolbox:
    # Creator guards (avoid re-creation on repeated runs)
    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Gene / individual / population
    toolbox.register("gene", lambda: random.randint(0, len(ALLOWED_PULSES) - 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, N_PULSES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation closes over cfg
    def _eval(ind):
        return (evaluate_indices(ind, cfg),)

    toolbox.register("evaluate", _eval)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(ALLOWED_PULSES) - 1, indpb=cfg.mut_indpb)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_k)

    # Optional parallel map
    if cfg.n_jobs and cfg.n_jobs > 0:
        import multiprocessing as mp
        pool = mp.Pool(processes=cfg.n_jobs)
        toolbox.register("map", pool.map)
        toolbox._pool = pool  # to close later

    # Seeding helper
    toolbox.seed_indices = seed_indices

    return toolbox


def run_ga_strong(N_PULSES: int,
                  seed_indices: List[int],
                  cfg: GAConfig) -> Tuple[List[int], List[float]]:
    """Stronger evolutionary loop using (μ + λ) strategy with elitism,
    adaptive mutpb, early stopping, and a Hall of Fame.
    Returns (best_indices, best_history).
    """
    toolbox = build_toolbox(N_PULSES, cfg, seed_indices)

    pop = toolbox.population(n=cfg.mu)
    # Ensure the seed sequence is present
    if toolbox.seed_indices is not None and len(pop) > 0:
        pop[0][:] = toolbox.seed_indices[:]

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    best_history: List[float] = []
    best_so_far = -np.inf
    gens_without_improve = 0

    mutpb = cfg.mutpb

    # Evaluate initial population
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid)) if hasattr(toolbox, "map") else list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid), **record)
    best_history.append(record["max"])  # type: ignore

    if record["max"] > best_so_far:  # type: ignore
        best_so_far = record["max"]  # type: ignore
        gens_without_improve = 0
    else:
        gens_without_improve += 1

    for gen in range(1, cfg.ngen + 1):
        # Variation
        offspring = tools.selTournament(pop, cfg.lambda_, tournsize=cfg.tournament_k)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < cfg.cxpb:
                tools.cxTwoPoint(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mutpb:
                tools.mutUniformInt(ind, low=0, up=len(ALLOWED_PULSES) - 1, indpb=cfg.mut_indpb)
                del ind.fitness.values

        # Evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid)) if hasattr(toolbox, "map") else list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # (μ + λ) selection (elitism implicitly included)
        pop = tools.selBest(pop + offspring, cfg.mu)

        # HOF / stats
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)
        best_history.append(record["max"])  # type: ignore

        # Adaptive mutation & early stopping
        mutpb *= cfg.mutpb_decay
        if record["max"] > best_so_far:  # type: ignore
            best_so_far = record["max"]  # type: ignore
            gens_without_improve = 0
        else:
            gens_without_improve += 1
            if gens_without_improve >= cfg.patience:
                break

    # Close pool if used
    if hasattr(toolbox, "_pool"):
        toolbox._pool.close()
        toolbox._pool.join()

    best_ind = list(hof[0])
    return best_ind, best_history


# -------------------------------------------------
# Post‑processing: random pulse‑drop pruning search
# -------------------------------------------------

def evaluate_sequence(indices: List[int], cfg: GAConfig) -> float:
    return evaluate_indices(indices, cfg)


def random_drop_prune(indices: List[int],
                      cfg: GAConfig,
                      trials: int = 300,
                      max_drop_per_trial: int = 3,
                      accept_tie: bool = True) -> List[int]:
    """Stochastic local search that tries randomly dropping a few pulses
    (1..max_drop_per_trial) each trial and keeps the change if fitness
    does not worsen (or improves). Encourages shorter sequences when
    combined with len_penalty/time_penalty in cfg.
    """
    current = indices[:]
    base_fit = evaluate_sequence(current, cfg)

    for _ in range(trials):
        if len(current) <= 1:
            break
        k = random.randint(1, min(max_drop_per_trial, len(current)-1))
        drop_positions = sorted(random.sample(range(len(current)), k), reverse=True)
        candidate = current[:]
        for pos in drop_positions:
            candidate.pop(pos)
        cand_fit = evaluate_sequence(candidate, cfg)
        if cand_fit > base_fit or (accept_tie and cand_fit == base_fit):
            current, base_fit = candidate, cand_fit
    return current


# -----------------
# Utility helpers
# -----------------

def load_seed_sequence(filepath: str) -> List[Tuple[int, int]]:
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    # Each line is like: [axis, delta_n]
    seq = [(eval(line)[0], eval(line)[1]) for line in lines]
    return seq


def save_sequence_with_times(indices: List[int], out_path: str) -> None:
    seq = sequence_from_indices(indices)
    with open(out_path, "w") as f:
        for item in seq:
            f.write(str(item) + "\n")

def save_config(cfg: GAConfig, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


# --------------
# Demo / script
# --------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Load XY seed like your original code
    seed_pairs = load_seed_sequence('sequence_XY.txt')
    N_PULSES = len(seed_pairs)
    seed_indices = [ALLOWED_PULSES.index(p) for p in seed_pairs]

    cfg = GAConfig(
        mol_num=1000,
        temps=(25e-6, 25e-6, 25e-6),
        ngen=20,
        mu=40,
        lambda_=40,
        cxpb=0.65,
        mutpb=0.35,
        tournament_k=3,
        n_jobs=0,              # set >0 to enable multiprocessing
        len_penalty=0.0,       # try 0.01 to gently prefer shorter sequences
        time_penalty=0.0,      # try e.g. 0.5 per second if pulse_time is in s
        mutpb_decay=0.985,
        mut_indpb=0.12,
        patience=12,
    )

    best_idx, history = run_ga_strong(N_PULSES, seed_indices, cfg)

    # Optional: prune by random dropping
    pruned_idx = random_drop_prune(best_idx, cfg, trials=400, max_drop_per_trial=3, accept_tie=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir("sequences/" + ts)
    save_sequence_with_times(best_idx, "sequences/" + ts + "/best_sequence.txt")
    save_sequence_with_times(pruned_idx, "sequences/" + ts + "/best_sequence_pruned.txt")
    save_config(cfg, "sequences/" + ts + "/config.json")

    print("Best (pre-prune) length:", len(best_idx))
    print("Best (post-prune) length:", len(pruned_idx))
    print("Done.")
