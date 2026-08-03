
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

Topology = Literal["small-world", "scale-free", "random"]

PAYOFF_MATRIX = {
    "A": {"A": 1.0, "B": 0.0, "C": 1.0},
    "B": {"A": 0.0, "B": 2.0, "C": -4.0},
    "C": {"A": 0.0, "B": 0.0, "C": 0.0},
}


@dataclass(frozen=True)
class SimulationConfig:
    num_nodes: int = 200
    steps: int = 2000
    sample_interval: int = 20
    control_interval: int = 60

    beta: float = 0.6
    initial_truth_probability: float = 0.5

    fixed_fact_checker_ratio: float = 0.25
    min_fact_checker_ratio: float = 0.05
    max_fact_checker_ratio: float = 0.50
    adaptive_gain: float = 0.70
    lasting_correction_probability: float = 0.70
    boundary_epsilon: float = 1e-12

    small_world_degree: int = 6
    small_world_rewiring_probability: float = 0.10
    scale_free_m: int = 3
    random_edge_probability: float = 0.03


@dataclass(frozen=True)
class SimulationEnvironment:
    graph: nx.Graph
    nodes: tuple[int, ...]
    num_nodes: int
    topology: Topology


@dataclass
class Trajectory:
    times: list[int]
    truthful_counts: list[int]
    fake_counts: list[int]
    fact_checker_ratios: list[float]

    @property
    def final_fake_count(self) -> int:
        return self.fake_counts[-1]

    @property
    def mean_fact_checker_budget(self) -> float:
        return float(np.mean(self.fact_checker_ratios))

    def efficiency(self, num_nodes: int) -> float:
        if len(self.times) < 2:
            return 1.0 - self.fake_counts[-1] / num_nodes

        auc_b = float(np.trapezoid(self.fake_counts, self.times))
        auc_max = float(num_nodes * (self.times[-1] - self.times[0]))
        if auc_max <= 0:
            return 1.0
        return 1.0 - auc_b / auc_max


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def build_environment(
    config: SimulationConfig,
    topology: Topology,
    seed: int,
) -> SimulationEnvironment:
    if topology == "small-world":
        graph = nx.watts_strogatz_graph(
            n=config.num_nodes,
            k=config.small_world_degree,
            p=config.small_world_rewiring_probability,
            seed=seed,
        )
    elif topology == "scale-free":
        graph = nx.barabasi_albert_graph(
            n=config.num_nodes,
            m=config.scale_free_m,
            seed=seed,
        )
    elif topology == "random":
        graph = nx.erdos_renyi_graph(
            n=config.num_nodes,
            p=config.random_edge_probability,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    nodes = tuple(graph.nodes())
    return SimulationEnvironment(
        graph=graph,
        nodes=nodes,
        num_nodes=len(nodes),
        topology=topology,
    )


def initialize_latent_opinions(
    env: SimulationEnvironment,
    config: SimulationConfig,
    seed: int,
) -> dict[int, str]:
    rng = random.Random(seed)
    return {
        node: (
            "A"
            if rng.random() < config.initial_truth_probability
            else "B"
        )
        for node in env.nodes
    }


def choose_random_fact_checkers(
    env: SimulationEnvironment,
    ratio: float,
    seed: int,
) -> set[int]:
    rng = random.Random(seed)
    count = math.floor(ratio * env.num_nodes)
    return set(rng.sample(list(env.nodes), count))


def effective_strategy(
    opinions: dict[int, str],
    fact_checkers: set[int],
    node: int,
) -> str:
    return "C" if node in fact_checkers else opinions[node]


def cumulative_payoff(
    env: SimulationEnvironment,
    opinions: dict[int, str],
    fact_checkers: set[int],
    node: int,
) -> float:
    node_strategy = effective_strategy(opinions, fact_checkers, node)
    return sum(
        PAYOFF_MATRIX[node_strategy][
            effective_strategy(opinions, fact_checkers, neighbor)
        ]
        for neighbor in env.graph.neighbors(node)
    )


def asynchronous_imitation_step(
    env: SimulationEnvironment,
    config: SimulationConfig,
    opinions: dict[int, str],
    fact_checkers: set[int],
    rng: random.Random,
) -> None:
    eligible_nodes = [
        node for node in env.nodes if node not in fact_checkers
    ]
    if not eligible_nodes:
        return

    node = rng.choice(eligible_nodes)
    eligible_neighbors = [
        neighbor
        for neighbor in env.graph.neighbors(node)
        if neighbor not in fact_checkers
    ]
    if not eligible_neighbors:
        return

    log_fitness = np.array(
        [
            config.beta
            * cumulative_payoff(
                env,
                opinions,
                fact_checkers,
                neighbor,
            )
            for neighbor in eligible_neighbors
        ],
        dtype=float,
    )
    weights = np.exp(log_fitness - np.max(log_fitness))
    total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0:
        return

    selected_neighbor = rng.choices(
        eligible_neighbors,
        weights=weights,
        k=1,
    )[0]
    opinions[node] = opinions[selected_neighbor]


def count_observable_states(
    env: SimulationEnvironment,
    opinions: dict[int, str],
    fact_checkers: set[int],
) -> tuple[int, int]:
    truthful_count = sum(
        1
        for node in env.nodes
        if node not in fact_checkers and opinions[node] == "A"
    )
    fake_count = sum(
        1
        for node in env.nodes
        if node not in fact_checkers and opinions[node] == "B"
    )
    return truthful_count, fake_count


def misinformation_prevalence(
    env: SimulationEnvironment,
    opinions: dict[int, str],
    fact_checkers: set[int],
) -> float:
    truthful_count, fake_count = count_observable_states(
        env,
        opinions,
        fact_checkers,
    )
    denominator = truthful_count + fake_count
    return fake_count / denominator if denominator else 0.0


def adaptive_fact_checker_ratio(
    env: SimulationEnvironment,
    config: SimulationConfig,
    opinions: dict[int, str],
    fact_checkers: set[int],
) -> float:
    prevalence = misinformation_prevalence(
        env,
        opinions,
        fact_checkers,
    )
    return clip(
        config.min_fact_checker_ratio
        + config.adaptive_gain * prevalence,
        config.min_fact_checker_ratio,
        config.max_fact_checker_ratio,
    )


def choose_boundary_fact_checkers(
    env: SimulationEnvironment,
    config: SimulationConfig,
    opinions: dict[int, str],
    current_fact_checkers: set[int],
    target_ratio: float,
) -> set[int]:
    target_count = math.floor(target_ratio * env.num_nodes)
    scores: list[tuple[float, int]] = []

    for node in env.nodes:
        if node in current_fact_checkers:
            continue

        fake_neighbors = 0
        truthful_neighbors = 0

        for neighbor in env.graph.neighbors(node):
            if neighbor in current_fact_checkers:
                continue
            if opinions[neighbor] == "B":
                fake_neighbors += 1
            else:
                truthful_neighbors += 1

        degree = env.graph.degree(node)
        score = (
            fake_neighbors * truthful_neighbors
        ) / ((degree + config.boundary_epsilon) ** 2)
        scores.append((score, node))

    if target_count > len(scores):
        raise RuntimeError(
            "Not enough non-fact-checker candidates for replacement."
        )

    scores.sort(key=lambda item: (-item[0], item[1]))
    return {node for _, node in scores[:target_count]}


def apply_lasting_correction(
    opinions: dict[int, str],
    removed_fact_checkers: set[int],
    probability: float,
    rng: random.Random,
) -> None:
    for node in removed_fact_checkers:
        if opinions[node] == "B" and rng.random() < probability:
            opinions[node] = "A"


def record_state(
    trajectory: Trajectory,
    time_step: int,
    env: SimulationEnvironment,
    opinions: dict[int, str],
    fact_checkers: set[int],
) -> None:
    truthful_count, fake_count = count_observable_states(
        env,
        opinions,
        fact_checkers,
    )
    trajectory.times.append(time_step)
    trajectory.truthful_counts.append(truthful_count)
    trajectory.fake_counts.append(fake_count)
    trajectory.fact_checker_ratios.append(
        len(fact_checkers) / env.num_nodes
    )


def run_static_baseline(
    env: SimulationEnvironment,
    config: SimulationConfig,
    initial_opinions: dict[int, str],
    fact_checker_seed: int,
    dynamics_seed: int,
) -> Trajectory:
    opinions = initial_opinions.copy()
    fact_checkers = choose_random_fact_checkers(
        env,
        config.fixed_fact_checker_ratio,
        fact_checker_seed,
    )
    dynamics_rng = random.Random(dynamics_seed)
    trajectory = Trajectory([], [], [], [])

    record_state(trajectory, 0, env, opinions, fact_checkers)

    for time_step in range(1, config.steps + 1):
        asynchronous_imitation_step(
            env,
            config,
            opinions,
            fact_checkers,
            dynamics_rng,
        )
        if time_step % config.sample_interval == 0:
            record_state(
                trajectory,
                time_step,
                env,
                opinions,
                fact_checkers,
            )

    return trajectory


def run_adaptive_model(
    env: SimulationEnvironment,
    config: SimulationConfig,
    initial_opinions: dict[int, str],
    dynamics_seed: int,
    correction_seed: int,
) -> Trajectory:
    opinions = initial_opinions.copy()
    fact_checkers: set[int] = set()
    dynamics_rng = random.Random(dynamics_seed)
    correction_rng = random.Random(correction_seed)
    trajectory = Trajectory([], [], [], [])

    target_ratio = adaptive_fact_checker_ratio(
        env,
        config,
        opinions,
        fact_checkers,
    )
    fact_checkers = choose_boundary_fact_checkers(
        env,
        config,
        opinions,
        fact_checkers,
        target_ratio,
    )
    record_state(trajectory, 0, env, opinions, fact_checkers)

    for time_step in range(1, config.steps + 1):
        asynchronous_imitation_step(
            env,
            config,
            opinions,
            fact_checkers,
            dynamics_rng,
        )

        if time_step % config.control_interval == 0:
            target_ratio = adaptive_fact_checker_ratio(
                env,
                config,
                opinions,
                fact_checkers,
            )
            new_fact_checkers = choose_boundary_fact_checkers(
                env,
                config,
                opinions,
                fact_checkers,
                target_ratio,
            )
            removed_fact_checkers = (
                fact_checkers - new_fact_checkers
            )
            apply_lasting_correction(
                opinions,
                removed_fact_checkers,
                config.lasting_correction_probability,
                correction_rng,
            )
            fact_checkers = new_fact_checkers

        if time_step % config.sample_interval == 0:
            record_state(
                trajectory,
                time_step,
                env,
                opinions,
                fact_checkers,
            )

    return trajectory


def run_paired_simulation(
    config: SimulationConfig,
    topology: Topology,
    seed: int,
) -> tuple[Trajectory, Trajectory]:
    env = build_environment(config, topology, seed)
    initial_opinions = initialize_latent_opinions(
        env,
        config,
        seed + 10_000,
    )

    baseline = run_static_baseline(
        env,
        config,
        initial_opinions,
        fact_checker_seed=seed + 20_000,
        dynamics_seed=seed + 30_000,
    )
    adaptive = run_adaptive_model(
        env,
        config,
        initial_opinions,
        dynamics_seed=seed + 40_000,
        correction_seed=seed + 50_000,
    )
    return baseline, adaptive


def plot_single_run(
    baseline: Trajectory,
    adaptive: Trajectory,
    topology: Topology,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    figure = plt.figure(figsize=(12, 5))

    axis1 = figure.add_subplot(1, 3, 1)
    axis1.plot(
        baseline.times,
        baseline.truthful_counts,
        label="A (static baseline)",
    )
    axis1.plot(
        baseline.times,
        baseline.fake_counts,
        label="B (static baseline)",
    )
    axis1.set_title("Static baseline: fixed random fact-checkers")
    axis1.set_xlabel("Time step")
    axis1.set_ylabel("Count among non-C")
    axis1.legend()
    axis1.grid(True, linewidth=0.3)

    axis2 = figure.add_subplot(1, 3, 2)
    axis2.plot(
        adaptive.times,
        adaptive.truthful_counts,
        label="A (adaptive)",
    )
    axis2.plot(
        adaptive.times,
        adaptive.fake_counts,
        label="B (adaptive)",
    )
    axis2.set_title("Adaptive density + boundary placement")
    axis2.set_xlabel("Time step")
    axis2.set_ylabel("Count among non-C")
    axis2.legend()
    axis2.grid(True, linewidth=0.3)

    axis3 = figure.add_subplot(1, 3, 3)
    axis3.plot(
        baseline.times,
        baseline.fact_checker_ratios,
        label="pC baseline",
    )
    axis3.plot(
        adaptive.times,
        adaptive.fact_checker_ratios,
        label="pC adaptive",
    )
    axis3.set_title(f"Fact-checker density ({topology})")
    axis3.set_xlabel("Time step")
    axis3.set_ylabel("pC")
    axis3.set_ylim(0.0, 0.6)
    axis3.legend()
    axis3.grid(True, linewidth=0.3)

    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def write_experiment_results(
    config: SimulationConfig,
    runs: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "topology",
        "seed",
        "strategy",
        "final_fake_count",
        "efficiency",
        "mean_fact_checker_budget",
    ]

    rows: list[dict[str, object]] = []

    for topology in ("small-world", "scale-free", "random"):
        for seed in range(runs):
            baseline, adaptive = run_paired_simulation(
                config,
                topology,
                seed,
            )
            for strategy, trajectory in (
                ("baseline", baseline),
                ("adaptive", adaptive),
            ):
                rows.append(
                    {
                        "topology": topology,
                        "seed": seed,
                        "strategy": strategy,
                        "final_fake_count": trajectory.final_fake_count,
                        "efficiency": trajectory.efficiency(
                            config.num_nodes
                        ),
                        "mean_fact_checker_budget": (
                            trajectory.mean_fact_checker_budget
                        ),
                    }
                )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    for topology in ("small-world", "scale-free", "random"):
        for strategy in ("baseline", "adaptive"):
            subset = [
                row
                for row in rows
                if row["topology"] == topology
                and row["strategy"] == strategy
            ]
            final_values = np.array(
                [row["final_fake_count"] for row in subset],
                dtype=float,
            )
            efficiency_values = np.array(
                [row["efficiency"] for row in subset],
                dtype=float,
            )
            budget_values = np.array(
                [
                    row["mean_fact_checker_budget"]
                    for row in subset
                ],
                dtype=float,
            )
            print(
                f"{topology:11s} {strategy:8s} | "
                f"final B={final_values.mean():.2f}±{final_values.std(ddof=1):.2f}, "
                f"efficiency={efficiency_values.mean():.3f}±"
                f"{efficiency_values.std(ddof=1):.3f}, "
                f"budget={budget_values.mean():.3f}±"
                f"{budget_values.std(ddof=1):.3f}"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-aligned implementation of adaptive boundary-aware "
            "fact-checker placement."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("single", "experiment"),
        default="single",
    )
    parser.add_argument(
        "--topology",
        choices=("small-world", "scale-free", "random"),
        default="small-world",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path in single mode or CSV path in experiment mode.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = SimulationConfig(steps=args.steps)

    if args.mode == "single":
        baseline, adaptive = run_paired_simulation(
            config,
            args.topology,
            args.seed,
        )
        plot_single_run(
            baseline,
            adaptive,
            args.topology,
            output_path=args.output,
            show=not args.no_show,
        )
        print(
            "Baseline:",
            {
                "final_B": baseline.final_fake_count,
                "efficiency": round(
                    baseline.efficiency(config.num_nodes), 4
                ),
                "mean_budget": round(
                    baseline.mean_fact_checker_budget, 4
                ),
            },
        )
        print(
            "Adaptive:",
            {
                "final_B": adaptive.final_fake_count,
                "efficiency": round(
                    adaptive.efficiency(config.num_nodes), 4
                ),
                "mean_budget": round(
                    adaptive.mean_fact_checker_budget, 4
                ),
            },
        )
    else:
        output_path = (
            args.output
            if args.output is not None
            else Path("results") / "paper_experiment_summary.csv"
        )
        write_experiment_results(
            config,
            runs=args.runs,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
