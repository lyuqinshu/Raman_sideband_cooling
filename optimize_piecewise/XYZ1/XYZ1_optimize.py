from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable, Optional, DefaultDict
from collections import defaultdict

import random
import numpy as np
from datetime import datetime
import json
from deap import base, creator, tools
import os
import RSC_sim
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.stats

# -----------------------------
# Problem definition / fitness
# -----------------------------

# ALLOWED_PULSES: List[Tuple[int, int]] = [
#     (0, -6), (0, -5), (0, -4), (0, -3), (0, -2),
#     (1, -6), (1, -5), (1, -4), (1, -3), (1, -2),
# ]

@dataclass
class GAConfig:
    mol_num: int = 1000
    temps: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    allowed_pulses: Tuple[Tuple[int, int], ...] = ((0, -6), (0, -5), (0, -4), (0, -3), (0, -2), (1, -6), (1, -5), (1, -4), (1, -3), (1, -2))
    ngen: int = 10
    mu: int = 40         
    lambda_: int = 20    
    cxpb: float = 0.65
    mutpb: float = 0.30
    tournament_k: int = 3
    # Fitness shaping
    len_penalty: float = 0.0
    time_penalty: float = 0.0
    # Adaptive mutation schedule
    mutpb_decay: float = 0.98
    mut_indpb: float = 0.1
    # Early stopping
    patience: int = 10
    min_len: int = 20
    max_len: int = 60
    p_insert: float = 0.20
    p_delete: float = 0.20
    p_modify: float = 0.60
    random_seed: int = 42

# -----------------------------
# Helpers
# -----------------------------

def cost_function(mol_list: Iterable) -> int:
    good = 0
    for mol in mol_list:
        if mol.n[0] <= 1 and mol.n[1] <= 1 and mol.n[2] <= 28 and mol.state == 1 and mol.spin == 0 and not mol.islost:
            good += 1
    return good

def sequence_from_indices(indices: List[int], cfg: GAConfig) -> List[Tuple[int, int, float]]:
    pulses = [cfg.allowed_pulses[i] for i in indices]
    return [[axis, dn, RSC_sim.pulse_time(axis, dn)] for axis, dn in pulses]
    
import ast

def load_seed_sequence(filepath: str) -> List[Tuple[int, int]]:
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    # safer than eval
    seq = [tuple(ast.literal_eval(line)) for line in lines]
    return [(int(a), int(dn)) for (a, dn) in seq]


# ------------- job worker (single molecule) -------------

def _eval_job(ind_idx: int, mol, pulse_sequence, optical_pumping: bool):
    """
    Worker: run one molecule through the sequence, return (ind_idx, updated_mol).
    """
    updated = RSC_sim.apply_raman_pulses_serial(mol, pulse_sequence, optical_pumping=optical_pumping, rng=None)
    return ind_idx, updated

# -----------------------------------
# GA with job-level parallel evaluation
# -----------------------------------

def build_toolbox(N_PULSES: int, cfg: GAConfig, seed_indices: Optional[List[int]] = None) -> base.Toolbox:
    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual() -> creator.Individual:
        L = random.randint(cfg.min_len, cfg.max_len)
        return creator.Individual([random.randint(0, len(cfg.allowed_pulses) - 1) for _ in range(L)])

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # We won’t use toolbox.evaluate/map for fitness; evaluation is batched per generation with jobs
    toolbox.register("mate", _cx_varlen_factory(cfg))
    toolbox.register("mutate", _mut_varlen_factory(cfg))
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_k)

    # Force serial map for any DEAP internals that call it
    toolbox.register("map", map)

    # Seed
    if seed_indices is not None:
        seed = creator.Individual(seed_indices[:])
        if len(seed) < cfg.min_len:
            seed.extend([random.randint(0, len(cfg.allowed_pulses)-1) for _ in range(cfg.min_len - len(seed))])
        if len(seed) > cfg.max_len:
            del seed[cfg.max_len:]
        toolbox.seed_individual = seed
    else:
        toolbox.seed_individual = None

    return toolbox

def _cx_varlen_factory(cfg: GAConfig):
    def cx_varlen(ind1, ind2):
        if len(ind1) < 2 or len(ind2) < 2:
            return ind1, ind2
        cx1 = random.randint(1, len(ind1)-1)
        cx2 = random.randint(1, len(ind2)-1)
        child1 = type(ind1)(ind1[:cx1] + ind2[cx2:])
        child2 = type(ind2)(ind2[:cx2] + ind1[cx1:])
        if len(child1) > cfg.max_len: del child1[cfg.max_len:]
        if len(child2) > cfg.max_len: del child2[cfg.max_len:]
        return child1, child2
    return cx_varlen

def _mut_varlen_factory(cfg: GAConfig):
    def mut_varlen(ind):
        if random.random() < cfg.p_modify and len(ind) > 0:
            i = random.randrange(len(ind))
            ind[i] = random.randint(0, len(cfg.allowed_pulses)-1)
        if random.random() < cfg.p_insert and len(ind) < cfg.max_len:
            i = random.randrange(len(ind)+1)
            ind.insert(i, random.randint(0, len(cfg.allowed_pulses)-1))
        if random.random() < cfg.p_delete and len(ind) > cfg.min_len:
            i = random.randrange(len(ind))
            del ind[i]
        return (ind,)
    return mut_varlen

# --------- Core: evaluate a batch of individuals via 1000-per-ind jobs ---------

from tqdm import tqdm
def _pairs_to_timed_sequence(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    return [(axis, dn, RSC_sim.pulse_time(axis, dn)) for axis, dn in pairs]

def _evaluate_individuals_via_jobs(individuals: List[List[int]],
                                   cfg: GAConfig,
                                   max_workers: Optional[int] = None,
                                   optical_pumping: bool = True) -> List[float]:
    """
    Creates cfg.mol_num jobs per individual (one molecule per job),
    runs them across a process pool with work-stealing,
    aggregates per-individual molecules, and returns fitness list aligned to 'individuals'.
    Shows a progress bar for completed jobs.
    """
    prefix_pairs = load_seed_sequence('XY_optimized.txt')  # if you actually want a fixed prefix
    prefix_seq = _pairs_to_timed_sequence(prefix_pairs)

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Pre-compute pulse sequences for all individuals
    seqs = [prefix_seq + sequence_from_indices(ind, cfg) for ind in individuals]

    # Prepare per-individual molecule storage
    from collections import defaultdict
    per_ind_mols: DefaultDict[int, List] = defaultdict(list)

    total_jobs = cfg.mol_num * len(individuals)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for ind_idx, seq in enumerate(seqs):
            mols = RSC_sim.initialize_thermal(list(cfg.temps), cfg.mol_num)
            for mol in mols:
                futures.append(ex.submit(_eval_job, ind_idx, mol, seq, optical_pumping))

        with tqdm(total=total_jobs, desc="Evaluating jobs", unit="mol", mininterval=10) as pbar:
            for fut in as_completed(futures):
                ind_idx, updated_mol = fut.result()
                per_ind_mols[ind_idx].append(updated_mol)
                pbar.update(1)

    # Compute fitness per individual
    fitnesses: List[float] = []
    for ind_idx, ind in enumerate(individuals):
        mols_done = per_ind_mols[ind_idx]
        score = float(cost_function(mols_done))
        if cfg.len_penalty:
            score -= cfg.len_penalty * len(ind)
        if cfg.time_penalty:
            total_t = sum(s[2] for s in seqs[ind_idx])
            score -= cfg.time_penalty * total_t
        fitnesses.append(score)

    return fitnesses


# --------------- GA main loop ----------------

def run_ga_strong(cfg: GAConfig,
                  seed_pairs,
                  EVAL_MAX_WORKERS: Optional[int] = None,
                  ) -> Tuple[List[int], List[float]]:
    N_PULSES = len(seed_pairs)
    seed_indices = [cfg.allowed_pulses.index(p) for p in seed_pairs]
    toolbox = build_toolbox(N_PULSES, cfg, seed_indices)
    pop = toolbox.population(n=cfg.mu)

    # Seed
    seed = getattr(toolbox, "seed_individual", None)
    if seed is None and seed_indices is not None:
        try:
            seed = type(pop[0])(seed_indices[:])
        except Exception:
            seed = list(seed_indices[:])

    if len(seed) < cfg.min_len:
        seed.extend([random.randint(0, len(cfg.allowed_pulses)-1) for _ in range(cfg.min_len - len(seed))])
    if len(seed) > cfg.max_len:
        del seed[cfg.max_len:]
    if len(pop) > 0:
        pop[0][:] = list(seed)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("std", scipy.stats.sem)
    stats.register("all", np.asarray)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    best_history: List[float] = []
    history = []
    best_so_far = -np.inf
    gens_without_improve = 0

    mutpb = cfg.mutpb

    # ---- Evaluate initial population via jobs ----
    print("Running generation ", 0)
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = _evaluate_individuals_via_jobs(invalid, cfg, max_workers=EVAL_MAX_WORKERS)
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = (fit,)
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid), **record)
    best_history.append(record["max"])
    history.append(record["all"])
    print("Initial eval: ", record['all'])

    if record["max"] > best_so_far:  
        best_so_far = record["max"]  
        gens_without_improve = 0
    else:
        gens_without_improve += 1

    # ---- Generational loop ----
    for gen in range(1, cfg.ngen + 1):
        print("Running generation ", gen)
        # Draw lambda_ samples from a tournament of size k
        parents = tools.selTournament(pop, cfg.lambda_, tournsize=cfg.tournament_k)
        offspring = list(map(toolbox.clone, parents))

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < cfg.cxpb:
                c1, c2 = toolbox.mate(offspring[i-1], offspring[i])
                offspring[i-1][:] = list(c1)
                offspring[i][:] = list(c2)
                if hasattr(offspring[i-1].fitness, 'values'):
                    del offspring[i-1].fitness.values
                if hasattr(offspring[i].fitness, 'values'):
                    del offspring[i].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                if hasattr(ind.fitness, 'values'):
                    del ind.fitness.values

        # Evaluate offspring via jobs (1000 * λ jobs)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = _evaluate_individuals_via_jobs(invalid, cfg, max_workers=EVAL_MAX_WORKERS)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = (fit,)

        # (μ + λ) selection with elitism
        pop = tools.selBest(pop + offspring, cfg.mu)

        for ind in pop:
            if hasattr(ind.fitness, "values"):
                del ind.fitness.values
        fits = _evaluate_individuals_via_jobs(pop, cfg, max_workers=EVAL_MAX_WORKERS)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = (fit,)

        # HOF / stats
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)
        best_history.append(record["max"])  # type: ignore
        history.append(record["all"])
        print("Gen ", gen, ": ", record['all'])

        # Adaptive mutation & early stopping
        mutpb *= cfg.mutpb_decay
        if record["max"] > best_so_far:  # type: ignore
            best_so_far = record["max"]  # type: ignore
            gens_without_improve = 0
        else:
            gens_without_improve += 1
            if gens_without_improve >= cfg.patience:
                break

    best_idx = list(hof[0])

    print("Best length: ", len(best_idx))
    print("Best history: ", best_history)
    print("Done.")

    return best_idx, history

# -----------------
# Utility helpers
# -----------------

def load_seed_sequence(filepath: str) -> List[Tuple[int, int]]:
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    seq = [(eval(line)[0], eval(line)[1]) for line in lines]
    return seq

def save_sequence_with_times(indices: List[int], cfg: GAConfig, out_path: str) -> None:
    seq = sequence_from_indices(indices, cfg)
    with open(out_path, "w") as f:
        for item in seq:
            f.write(str(item) + "\n")

def save_config(cfg: GAConfig, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

def run_ga_master(cfg: GAConfig, seed_pairs, 
                  EVAL_MAX_WORKERS=None, 
                  file_dir = "sequences/"):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    
    best_idx, history = run_ga_strong(cfg, seed_pairs, EVAL_MAX_WORKERS)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_dir = file_dir + ts
    os.makedirs(file_dir, exist_ok=True)
    save_sequence_with_times(best_idx, cfg, file_dir + "/best_sequence.txt")
    save_config(cfg, file_dir + "/config.json")
    with open(file_dir + "/history.txt", "w") as f:
        for val in history:
            f.write(f"{val}\n")

    with open(file_dir + "/allowed_pulses.txt", "w") as f:
        for p in cfg.allowed_pulses:
            f.write(f"{p}\n")

# --------------
# Demo / script
# --------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    

    cfg = GAConfig(
        mol_num=1000,
        temps=(25e-6, 25e-6, 25e-6),
        allowed_pulses=((0, -2), (0, -1), (1, -2), (1, -1), (2, -9), (2, -8), (2, -7), (2, -6), (2, -5), (2, -4)),
        ngen=2,
        mu=10, # population size
        lambda_=5, # number of selected parents after tournament
        cxpb=0.65,
        mutpb=0.35,
        tournament_k=3, # tournament size
        len_penalty=0.5,
        time_penalty=0.0,
        mutpb_decay=0.985,
        mut_indpb=0.12,
        patience=6,
        min_len=20,
        max_len=80,
        p_insert=0.20,
        p_delete=0.20,
        p_modify=0.60,
        random_seed=42,
    )

    seed_pairs = load_seed_sequence('sequence_XYZ1.txt')

    run_ga_master(cfg, seed_pairs, EVAL_MAX_WORKERS=None, file_dir = "sequences/")
