import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationConfig:
    num_nodes: int = 200
    neighbors_per_node: int = 6
    rewiring_probability: float = 0.1
    network_seed: int = 1

    beta: float = 0.6
    steps: int = 2000
    control_interval: int = 60
    sample_interval: int = 20

    fixed_fact_checker_ratio: float = 0.25
    min_fact_checker_ratio: float = 0.05
    max_fact_checker_ratio: float = 0.35
    adaptive_gain: float = 0.70

@dataclass(frozen=True)
class SimulationEnvironment:
    graph: nx.Graph
    nodes: tuple[int, ...]
    num_nodes: int

def build_environment(
    config: SimulationConfig,
) -> SimulationEnvironment:
    graph = nx.watts_strogatz_graph(
        n=config.num_nodes,
        k=config.neighbors_per_node,
        p=config.rewiring_probability,
        seed=config.network_seed,
    )

    nodes = tuple(graph.nodes())

    return SimulationEnvironment(
        graph=graph,
        nodes=nodes,
        num_nodes=len(nodes),
    )

CONFIG = SimulationConfig()
ENV = build_environment(CONFIG)

PAYOFF_MATRIX = {
    "A": {"A": 1, "B": 0, "C": 1},
    "B": {"A": 0, "B": 2, "C": -4},
    "C": {"A": 0, "B": 0, "C": 0},
}

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def strategy(opinion, C_set, u):
    return "C" if u in C_set else opinion[u]

def initialize(
    env: SimulationEnvironment,
    seed=7,
    pC0=CONFIG.fixed_fact_checker_ratio,
    mode="degree",
):
    rng = random.Random(seed)

    opinion = {
        u: ("A" if rng.random() < 0.5 else "B")
        for u in env.nodes
    }

    nC = int(round(pC0 * env.num_nodes))

    if mode == "degree":
        degrees = dict(env.graph.degree())

        ranked = sorted(
            env.nodes,
            key=lambda u: degrees[u],
            reverse=True,
        )

        C_set = set(ranked[:nC])

    else:
        raise ValueError(f"Unknown placement mode: {mode}")

    return opinion, C_set

def avg_payoff(
    env: SimulationEnvironment,
    opinion,
    C_set,
    u,
):
    su = strategy(opinion, C_set, u)

    deg_u = env.graph.degree(u)
    if deg_u == 0:
        return 0.0

    total = 0.0

    for v in env.graph.neighbors(u):
        sv = strategy(opinion, C_set, v)
        total += PAYOFF_MATRIX[su][sv]

    return total / deg_u


def fitness(
    env: SimulationEnvironment,
    config: SimulationConfig,
    opinion,
    C_set,
    u,
):
    payoff_value = avg_payoff(
        env,
        opinion,
        C_set,
        u,
    )

    return math.exp(
        config.beta * payoff_value
    )

def choose_C_boundary(opinion, C_set, pC_current):
    """
    Place sanctioners on the boundary of fake-news spread:
    score(v) = (#B neighbors) * (#non-B neighbors)
    """
    nC = int(round(pC_current * ENV.num_nodes))
    scores = []

    for v in ENV.nodes:
        nb = 0
        nnonb = 0
        for u in ENV.graph.neighbors(v):
            su = strategy(opinion, C_set, u)
            if su == "B":
                nb += 1
            else:
                nnonb += 1
        scores.append((nb * nnonb, v))

    scores.sort(reverse=True, key=lambda x: x[0])
    return set(v for _, v in scores[:nC])


def compute_pC_from_B(
    env: SimulationEnvironment,
    config: SimulationConfig,
    opinion,
    C_set,
):
    """
    Increase the fact-checker ratio when fake-news prevalence rises.
    """
    viable_nodes = [
        u
        for u in env.nodes
        if u not in C_set
    ]

    if not viable_nodes:
        return config.max_fact_checker_ratio

    fake_news_count = sum(
        1
        for u in viable_nodes
        if opinion[u] == "B"
    )

    fake_news_ratio = (
        fake_news_count / len(viable_nodes)
    )

    new_ratio = (
        config.min_fact_checker_ratio
        + config.adaptive_gain * fake_news_ratio
    )

    return clip(
        new_ratio,
        config.min_fact_checker_ratio,
        config.max_fact_checker_ratio,
    )


def step_async(
    env: SimulationEnvironment,
    config: SimulationConfig,
    opinion,
    C_set,
    rng,
):
    u = rng.choice(env.nodes)

    if u in C_set:
        return

    neighbors = list(env.graph.neighbors(u))

    if not neighbors:
        return

    weights = np.array(
        [
            fitness(
                env,
                config,
                opinion,
                C_set,
                neighbor,
            )
            for neighbor in neighbors
        ],
        dtype=float,
    )

    if weights.sum() <= 0:
        return

    selected_neighbor = rng.choices(
        neighbors,
        weights=weights,
        k=1,
    )[0]

    if selected_neighbor in C_set:
        return

    opinion[u] = opinion[selected_neighbor]

def run_baseline_targeted(seed_init=7, seed_run=11, placement="degree"):

    opinion, C_set = initialize(
        ENV,
        seed=seed_init,
        pC0=CONFIG.fixed_fact_checker_ratio,
        mode=placement,
    )
    rng = random.Random(seed_run)

    histA, histB, hist_pC = [], [], []

    for t in range(CONFIG.steps):
        step_async(
            ENV,
            CONFIG,
            opinion,
            C_set,
            rng,
        )

        if t % CONFIG.sample_interval == 0:
            A = sum(
                1
                for u in ENV.nodes
                if u not in C_set and opinion[u] == "A"
            )

            B = sum(
                1
                for u in ENV.nodes
                if u not in C_set and opinion[u] == "B"
            )

            histA.append(A)
            histB.append(B)
            hist_pC.append(len(C_set) / ENV.num_nodes)

    return histA, histB, hist_pC


def run_upgrade(seed_init=7, seed_run=11, baseline_placement="degree"):
    opinion, C_set = initialize(
        ENV,
        seed=seed_init,
        pC0=CONFIG.fixed_fact_checker_ratio,
        mode=baseline_placement,
    )
    rng = random.Random(seed_run)

    histA, histB, hist_pC = [], [], []

    pC_current = CONFIG.fixed_fact_checker_ratio

    for t in range(CONFIG.steps):
        if t % CONFIG.control_interval == 0:
            pC_current = compute_pC_from_B(
                ENV,
                CONFIG,
                opinion,
                C_set,
            )
            C_set = choose_C_boundary(opinion, C_set, pC_current)

        step_async(
            ENV,
            CONFIG,
            opinion,
            C_set,
            rng,
        )

        if t % CONFIG.sample_interval == 0:
            A = sum(
                1
                for u in ENV.nodes
                if u not in C_set and opinion[u] == "A"
            )

            B = sum(
                1
                for u in ENV.nodes
                if u not in C_set and opinion[u] == "B"
            )

            histA.append(A)
            histB.append(B)
            hist_pC.append(
                len(C_set) / ENV.num_nodes
            )

    return histA, histB, hist_pC


PLACEMENT = "degree"  

A_base, B_base, p_base = run_baseline_targeted(placement=PLACEMENT)
A_up,   B_up,   p_up   = run_upgrade(baseline_placement=PLACEMENT)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(A_base, label="A (baseline-targeted)")
plt.plot(B_base, label="B (baseline-targeted)")
plt.title(f"Baseline: Targeted fixed C ({PLACEMENT})")
plt.xlabel("Time")
plt.ylabel("Count among non-C")
plt.legend()
plt.grid(True, linewidth=0.3)

plt.subplot(1, 3, 2)
plt.plot(A_up, label="A (upgrade)")
plt.plot(B_up, label="B (upgrade)")
plt.title("UPGRADE: pC(t) + boundary placement")
plt.xlabel("Time")
plt.ylabel("Count among non-C")
plt.legend()
plt.grid(True, linewidth=0.3)

plt.subplot(1, 3, 3)
plt.plot(p_base, label="pC baseline (fixed)")
plt.plot(p_up, label="pC upgrade (adaptive)")
plt.title("pC(t) comparison")
plt.xlabel("Time")
plt.ylabel("pC")
plt.ylim(0, 0.6)
plt.legend()
plt.grid(True, linewidth=0.3)

plt.tight_layout()
plt.show()
