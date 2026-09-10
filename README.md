# Moebius

State-based reinforcement-learning experiments for robot manipulation. Phase 1
is a Soft Actor-Critic (SAC) baseline on one MetaWorld task; Phase 2 adds the
core PEARL latent-task inference implementation.


## Included in Phase 1

- MetaWorld `MT1` wrapper with Gymnasium's `reset`/`step` API.
- Squashed Gaussian actor, twin Q critics, entropy temperature tuning, and
  Polyak-averaged target critics.
- Replay buffer, training/evaluation loop, checkpoints, and TensorBoard logs.
- Default task: `reach-v3`. Switch to another ML1 task using `--env-name`.

## Included in Phase 2 — PEARL core

- A seeded ML1 goal-task distribution with disjoint meta-train/meta-test task
  seeds. The task identifier is used only to organize replay data; it is never
  supplied to the agent.
- A task-specific replay buffer for every meta-training task. Each update draws
  both a context set and an independent RL batch from the same task.
- A permutation-invariant probabilistic context encoder
  `q(z | {(s, a, r, s')})`. Its per-transition Gaussian factors are combined
  with a standard-normal prior, then a latent `z` is sampled.
- SAC actor and twin critics conditioned on `z`: `pi(a | s, z)` and
  `Q(s, a, z)`, with critic-driven encoder learning and KL regularization.
- Adaptation evaluation that logs return and success before context and after
  each context-collection episode on held-out goal configurations.

## Setup

Use Python 3.10+ and install the package in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

MetaWorld is installed from its maintained Git repository, and its MuJoCo
backend may need platform-specific system libraries. Refer to the MetaWorld
installation documentation if its installation fails.

## Train

Run a short smoke experiment first:

```bash
moebius-train --total-steps 2000 --start-steps 500 --eval-interval 1000
```

Then start the standard baseline run:

```bash
moebius-train --env-name reach-v3 --total-steps 250000
```

## Train PEARL on a simple task distribution

This is intentionally the first PEARL milestone: the task differs by an ML1
goal configuration, rather than by object mass/friction or pick-and-place.
The agent must infer the configuration from transition context, then condition
its policy on the inferred latent.

```bash
moebius-train --algorithm pearl --env-name reach-v3 \
  --num-train-tasks 20 --num-eval-tasks 10 \
  --latent-dim 5 --context-size 32 --meta-batch-size 8 \
  --total-steps 250000
```

For a quick wiring check, shrink both training and meta-batch parameters so
each task receives data early:

```bash
moebius-train --algorithm pearl --total-steps 5000 --start-steps 1000 \
  --num-train-tasks 4 --num-eval-tasks 2 --meta-batch-size 2 \
  --batch-size 32 --context-size 16 --eval-interval 2500
```

View training and evaluation metrics:

```bash
tensorboard --logdir runs
```

For PEARL, inspect `loss/kl`, `stats/z_mean`, `stats/z_std`, and especially
`eval/success_after_context_0`, `eval/success_after_context_1`, and so on. An
improving held-out success curve as context episodes accumulate is the Phase 2
adaptation signal.

## Record a trained policy

Record deterministic evaluation episodes from a saved SAC checkpoint:

```bash
moebius-record --checkpoint checkpoints/reach-v3_sac_seed1.pt --episodes 3
```

Record PEARL's adaptation on one held-out task instead:

```bash
moebius-record --checkpoint checkpoints/reach-v3_pearl_seed1.pt \
  --task-index 0 --adaptation-episodes 3
```

This produces one stochastic context-collection video per adaptation episode,
followed by a deterministic `*_adapted.mp4` rollout using the inferred task
latent. Omit `--adaptation-episodes` to use the value saved in the checkpoint.
Use `--task-index` to select another held-out task configuration.

Videos are written to `videos/`. Use `--video-dir` or `--fps` to change the
output folder or frame rate. Recording uses a top-down camera by default to
keep the object visible. Other available camera angles include:

```bash
moebius-record --checkpoint checkpoints/reach-v3_sac_seed1.pt --camera-name corner2
```

The final checkpoint is saved to `checkpoints/`. Tests that do not require a
MuJoCo installation can be run with:

```bash
pytest
```
