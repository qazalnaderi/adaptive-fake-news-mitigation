import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


G = nx.watts_strogatz_graph(n=120, k=6, p=0.1, seed=1)
nodes = list(G.nodes())
N = len(nodes)


beta = 0.5
steps = 4000
T = 50

payoff = {
    "A": {"A": 1, "B": 0, "C": 1},
    "B": {"A": 0, "B": 2, "C": -4},
    "C": {"A": 0, "B": 0, "C": 0},
}

pC_fixed = 0.20

p_min = 0.05
p_max = 0.35
k_gain = 0.90


def clip(x, lo, hi):
    return max(lo, min(hi, x))

def strategy(opinion, C_set, u):
    return "C" if u in C_set else opinion[u]

def initialize(seed=7, pC0=0.2, mode="degree"):
    rng = random.Random(seed)

    opinion = {u: ("A" if rng.random() < 0.5 else "B") for u in nodes}
    # 0.2*120=24
    nC = int(round(pC0 * N))

    if mode == "degree":
        deg = dict(G.degree())
        ranked = sorted(nodes, key=lambda u: deg[u], reverse=True)
        C_set = set(ranked[:nC])

    else:
        raise ValueError("Unknown mode")

    return opinion, C_set


def avg_payoff(opinion, C_set, u):
    su = strategy(opinion, C_set, u)
    deg_u = G.degree(u)
    if deg_u == 0:
        return 0.0

    total = 0.0
    for v in G.neighbors(u):
        sv = strategy(opinion, C_set, v)
        total += payoff[su][sv]
    return total / deg_u

def fitness(opinion, C_set, u):
    return math.exp(beta * avg_payoff(opinion, C_set, u))

def choose_C_boundary(opinion, C_set, pC_current):
    """
    Place sanctioners on the boundary of fake-news spread:
    score(v) = (#B neighbors) * (#non-B neighbors)
    """
    nC = int(round(pC_current * N))
    scores = []

    for v in nodes:
        nb = 0
        nnonb = 0
        for u in G.neighbors(v):
            su = strategy(opinion, C_set, u)
            if su == "B":
                nb += 1
            else:
                nnonb += 1
        scores.append((nb * nnonb, v))

    scores.sort(reverse=True, key=lambda x: x[0])
    return set(v for _, v in scores[:nC])

def compute_pC_from_B(opinion, C_set):
    """
    Increase pC when fake news level is higher.
    """
    viable = [u for u in nodes if u not in C_set]
    if not viable:
        return p_max

    B_count = sum(1 for u in viable if opinion[u] == "B")
    B_frac = B_count / len(viable)

    pC_new = p_min + k_gain * B_frac
    return clip(pC_new, p_min, p_max)


def step_async(opinion, C_set, rng):
    u = rng.choice(nodes)
    if u in C_set:
        return

    neigh = list(G.neighbors(u))
    if not neigh:
        return

    weights = np.array([fitness(opinion, C_set, v) for v in neigh], dtype=float)
    if weights.sum() <= 0:
        return

    v = rng.choices(neigh, weights=weights, k=1)[0]
    if v in C_set:
        return

    opinion[u] = opinion[v]


def run_baseline_targeted(seed_init=7, seed_run=11, placement="degree"):
    opinion, C_set = initialize(seed=seed_init, pC0=pC_fixed, mode=placement)
    rng = random.Random(seed_run)

    histA, histB, hist_pC = [], [], []

    for t in range(steps):
        step_async(opinion, C_set, rng)

        if t % 20 == 0:
            A = sum(1 for u in nodes if (u not in C_set and opinion[u] == "A"))
            B = sum(1 for u in nodes if (u not in C_set and opinion[u] == "B"))
            histA.append(A)
            histB.append(B)
            hist_pC.append(len(C_set) / N)

    return histA, histB, hist_pC


def run_upgrade(seed_init=7, seed_run=11, baseline_placement="degree"):
    opinion, C_set = initialize(seed=seed_init, pC0=pC_fixed, mode=baseline_placement)
    rng = random.Random(seed_run)

    histA, histB, hist_pC = [], [], []

    pC_current = pC_fixed

    for t in range(steps):
        if t % T == 0:
            pC_current = compute_pC_from_B(opinion, C_set)
            C_set = choose_C_boundary(opinion, C_set, pC_current)

        step_async(opinion, C_set, rng)

        if t % 20 == 0:
            A = sum(1 for u in nodes if (u not in C_set and opinion[u] == "A"))
            B = sum(1 for u in nodes if (u not in C_set and opinion[u] == "B"))
            histA.append(A)
            histB.append(B)
            hist_pC.append(len(C_set) / N)

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
