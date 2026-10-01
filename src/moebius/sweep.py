"""Measure whether object shape and hidden properties call for different behavior.

For every combination of gripper force limit, shape, and low/high values of
each hidden property, run every scripted expert variant for a few episodes
and record its success rate. The key output is the *adaptation gap*: success
when the best variant is picked separately for each object, minus success of
the single best variant used for all objects. A gap near zero means one fixed
behavior already handles every object, so a task-inference latent has
nothing to add.

Run with ``moebius-sweep`` (see ``--help``); results are written as CSV and
summarised on stdout.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from moebius.envs import MetaWorldSingleTaskEnv, ObjectPropertyBounds, TaskSpec
from moebius.experts import EXPERT_VARIANTS, PickPlaceExpert


def parse_args() -> argparse.Namespace:
    bounds = ObjectPropertyBounds()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapes", nargs="+", default=list(bounds.shapes))
    parser.add_argument("--variants", nargs="+", default=list(EXPERT_VARIANTS))
    parser.add_argument("--masses", nargs="+", type=float, default=[bounds.mass[0], 0.75, bounds.mass[1]])
    parser.add_argument(
        "--finger-frictions", nargs="+", type=float, default=[bounds.finger_friction[0], 1.0, bounds.finger_friction[1]]
    )
    parser.add_argument("--table-frictions", nargs="+", type=float, default=list(bounds.table_friction))
    parser.add_argument(
        "--force-limits",
        nargs="+",
        default=["none", "40", "20", "10"],
        help="Gripper force limits per finger in newtons; 'none' is MetaWorld's unlimited servo.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per setting and variant.")
    parser.add_argument(
        "--max-steps",
        nargs="+",
        type=int,
        default=[500],
        help="Episode lengths to evaluate. Each episode runs once at the longest length and success is read at "
        "every shorter one (a truncated episode is identical up to its cut-off). Shorter episodes make slow, "
        "careful handling costly, so the best behavior can differ between objects.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/sweep.csv"))
    return parser.parse_args()


def run_episode(
    environment: MetaWorldSingleTaskEnv, expert: PickPlaceExpert, lengths: list[int]
) -> tuple[dict[int, bool], float]:
    """Return ({length: success at that step}, mean pad-object normal force while gripping).

    Success at step ``length`` is what an episode truncated at ``length``
    would report as its final-step success.
    """
    observation, _ = environment.reset()
    terminated = truncated = False
    forces = []
    success_at: dict[int, bool] = {}
    step = 0
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = environment.step(expert.get_action(observation))
        step += 1
        if step in lengths:
            success_at[step] = bool(info.get("success", False))
        force = environment.grip_force()
        if force > 0:
            forces.append(force)
    return success_at, float(np.mean(forces)) if forces else 0.0


def sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    bounds = ObjectPropertyBounds()
    size = {
        "half_width": float(np.mean(bounds.half_width)),
        "half_height": float(np.mean(bounds.half_height)),
        "half_length": float(np.mean(bounds.bar_half_length)),
    }
    force_limits = [None if value.lower() == "none" else float(value) for value in args.force_limits]
    lengths = sorted(set(args.max_steps))
    rows: list[dict[str, object]] = []
    for force_limit in force_limits:
        environment = MetaWorldSingleTaskEnv(
            "pick-place-v3", args.seed, max_episode_steps=lengths[-1], gripper_force_limit=force_limit
        )
        try:
            grid = itertools.product(args.shapes, args.masses, args.finger_frictions, args.table_frictions)
            for shape, mass, finger_friction, table_friction in grid:
                task = TaskSpec(
                    0,
                    shape=shape,
                    object_half_width=size["half_width"],
                    object_half_height=size["half_width"] if shape == "sphere" else size["half_height"],
                    object_half_length=size["half_length"] if shape == "bar" else None,
                    object_mass=mass,
                    object_table_friction=table_friction,
                    object_finger_friction=finger_friction,
                )
                environment.set_task_properties(task)
                for variant in args.variants:
                    results = [
                        run_episode(environment, EXPERT_VARIANTS[variant], lengths) for _ in range(args.episodes)
                    ]
                    grip_forces = [force for _, force in results if force > 0]
                    half = args.episodes // 2
                    for length in lengths:
                        outcomes = [success_at[length] for success_at, _ in results]
                        row = {
                            "force_limit": "none" if force_limit is None else force_limit,
                            "max_steps": length,
                            "shape": shape,
                            "mass": mass,
                            "finger_friction": finger_friction,
                            "table_friction": table_friction,
                            "variant": variant,
                            "successes": sum(outcomes),
                            "episodes": args.episodes,
                            # Split halves for the held-out gap (see summarize).
                            "first_half_successes": sum(outcomes[:half]),
                            "first_half_episodes": half,
                            "mean_grip_force": float(np.mean(grip_forces)) if grip_forces else 0.0,
                        }
                        rows.append(row)
                    print(
                        f"force={row['force_limit']!s:>5} {shape:>8} mass={mass:<5g} finger_mu={finger_friction:<4g} "
                        f"table_mu={table_friction:<4g} {variant:>14}: "
                        + " ".join(
                            f"{length} steps {sum(r[0][length] for r in results)}/{args.episodes}" for length in lengths
                        )
                        + f" grip={row['mean_grip_force']:.1f} N",
                        flush=True,
                    )
        finally:
            environment.close()
    return rows


def held_out_gap(rows: list[dict[str, object]]) -> tuple[float, float, float] | None:
    """Adaptation gap with variant choices made on one half of the episodes.

    Picking the best of several noisy success estimates per object is biased
    upward, so the in-sample gap is positive even when one variant is truly
    best everywhere. Here both the single best variant and the per-object best
    variants are chosen on one half of the episodes and scored on the other
    half (then the halves swap). Returns (per-object success, best single
    success, gap), or None with fewer than two episodes per setting.
    """
    if any(row["first_half_episodes"] < 1 or row["episodes"] - row["first_half_episodes"] < 1 for row in rows):
        return None
    halves: dict[tuple, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        key = (row["shape"], row["mass"], row["finger_friction"], row["table_friction"])
        first = row["first_half_successes"] / row["first_half_episodes"]
        second = (row["successes"] - row["first_half_successes"]) / (row["episodes"] - row["first_half_episodes"])
        halves[key][row["variant"]] = (first, second)
    variants = sorted({variant for per_setting in halves.values() for variant in per_setting})
    results = []
    for choose, score in ((0, 1), (1, 0)):
        fixed = max(variants, key=lambda v: np.mean([h[v][choose] for h in halves.values()]))
        fixed_score = np.mean([h[fixed][score] for h in halves.values()])
        # Ties go to the single best variant, so noise alone cannot create a gap.
        per_object = np.mean(
            [
                h[max(h, key=lambda v: (h[v][choose], v == fixed))][score]
                for h in halves.values()
            ]
        )
        results.append((per_object, fixed_score))
    per_object, fixed_score = np.mean(results, axis=0)
    return float(per_object), float(fixed_score), float(per_object - fixed_score)


def summarize(rows: list[dict[str, object]]) -> None:
    """Print, per force limit and episode length, the adaptation gap and where success depends on the object."""
    by_force: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_force[(row["force_limit"], row["max_steps"])].append(row)
    for (force_limit, max_steps), force_rows in by_force.items():
        rate: dict[tuple, dict[str, float]] = defaultdict(dict)
        grip = []
        for row in force_rows:
            key = (row["shape"], row["mass"], row["finger_friction"], row["table_friction"])
            rate[key][row["variant"]] = row["successes"] / row["episodes"]
            if row["mean_grip_force"]:
                grip.append(row["mean_grip_force"])
        variants = sorted({variant for per_setting in rate.values() for variant in per_setting})
        fixed = {variant: np.mean([per_setting[variant] for per_setting in rate.values()]) for variant in variants}
        best_fixed = max(fixed, key=fixed.get)
        oracle = np.mean([max(per_setting.values()) for per_setting in rate.values()])
        print(f"\n=== gripper force limit: {force_limit}, {max_steps} steps (mean grip force while holding: "
              f"{np.mean(grip) if grip else 0:.1f} N) ===")
        print("success of each variant over all objects: " + ", ".join(f"{v} {fixed[v]:.0%}" for v in variants))
        print(f"best single variant: {best_fixed} ({fixed[best_fixed]:.0%}); best variant per object: {oracle:.0%}")
        print(f"in-sample adaptation gap: {oracle - fixed[best_fixed]:+.0%} (optimistic: best of noisy estimates)")
        held_out = held_out_gap(force_rows)
        if held_out is None:
            print("HELD-OUT ADAPTATION GAP: needs --episodes 2 or more")
        else:
            per_object, single, gap = held_out
            print(
                f"HELD-OUT ADAPTATION GAP: {gap:+.0%} (per-object choice {per_object:.0%} vs single best "
                f"{single:.0%}; near 0 = one behavior handles every object)"
            )
        for index, name in enumerate(("shape", "mass", "finger friction", "table friction")):
            levels = sorted({key[index] for key in rate}, key=str)
            summary = ", ".join(
                f"{level}: {np.mean([max(v.values()) for k, v in rate.items() if k[index] == level]):.0%}"
                for level in levels
            )
            print(f"  best-variant success by {name}: {summary}")
        shapes = sorted({key[0] for key in rate})
        print("  variant success by shape: " + " | ".join(
            f"{shape}: " + " ".join(
                f"{v} {np.mean([p[v] for k, p in rate.items() if k[0] == shape]):.0%}" for v in variants
            )
            for shape in shapes
        ))
        winners = defaultdict(int)
        for per_setting in rate.values():
            top = max(per_setting.values())
            if top > 0:
                for variant, value in per_setting.items():
                    if value == top:
                        winners[variant] += 1
        print("  settings where each variant is (joint) best: " + ", ".join(f"{v} {winners[v]}" for v in variants))


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.variants) - set(EXPERT_VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {sorted(EXPERT_VARIANTS)}.")
    if any(not 1 <= length <= 500 for length in args.max_steps):
        raise SystemExit("--max-steps values must be between 1 and 500 (MetaWorld's episode limit).")
    rows = sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.output}")
    summarize(rows)


if __name__ == "__main__":
    main()
