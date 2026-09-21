# Moebius

State-based reinforcement-learning experiments for robot manipulation. Phase 1
is a Soft Actor-Critic (SAC) baseline, Phase 2 adds PEARL latent-task
inference, Phase 3 applies that pipeline to MetaWorld pick-and-place, and
Phase 4 introduces hidden object dynamics.


## Included in Phase 1

- MetaWorld `MT1` wrapper with Gymnasium's `reset`/`step` API.
- Squashed Gaussian actor, twin Q critics, entropy temperature tuning, and
  Polyak-averaged target critics.
- Replay buffer, training/evaluation loop, checkpoints, and TensorBoard logs.

## Included in Phase 2 — PEARL core

- A seeded MT1 reset-configuration distribution with disjoint meta-train/
  meta-test task seeds. The task identifier is used only to organize replay
  data; it is never supplied to the agent.
- A task-specific replay buffer for every meta-training task. Each update draws
  both a context set and an independent RL batch from the same task.
- A permutation-invariant probabilistic context encoder
  `q(z | {(s, a, r, s')})`. Its per-transition Gaussian factors are combined
  with a standard-normal prior, then a latent `z` is sampled.
- SAC actor and twin critics conditioned on `z`: `pi(a | s, z)` and
  `Q(s, a, z)`, with critic-driven encoder learning and KL regularization.
- Adaptation evaluation that logs return and success before context and after
  each context-collection episode on held-out goal configurations.

## Included in Phase 3 — Pick-and-place

- `pick-place-v3` is now the default environment for both SAC and PEARL.
  Switch to another MT1 task using `--env-name`.
- Each seeded PEARL task reset selects a reproducible pick-and-place
  configuration. This varies the object's starting placement and the target
  configuration; the state observation continues to expose the robot, object,
  and goal state, while the replay-buffer task identifier is never an input.
- Demonstration-augmented SAC can collect successful scripted-expert rollouts
  across the training reset distribution, behavior-clone the actor, then mix
  permanent demonstrations with online experience during SAC updates.

## Included in Phase 4 — Hidden object properties

- Each PEARL task fixes an unobserved object mass scale and sliding-friction
  scale. The default train/test range is 0.5–1.5 times MetaWorld's nominal
  value, with independently sampled and disjoint train/test parameter pairs.
- Object and goal positions are still randomized on every reset within each
  task. This ensures that PEARL must infer dynamics from transitions rather
  than using a visible configuration as a task label.
- Before online training, MetaWorld's scripted pick-and-place expert collects
  demonstrations for every training dynamics task. PEARL behavior-clones both
  its actor and context encoder using chronological context windows of lengths
  1, 4, 8, 16, or 32, plus explicit empty-context (`z = 0`) training. It then
  uses persistent per-task demonstrations in mixed online/offline updates with
  a cloning-loss floor instead of removing the expert anchor entirely.

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

Then start the Phase 3 demonstration-augmented SAC run:

```bash
moebius-train --algorithm sac --total-steps 1000000 \
  --demo-episodes 100 --bc-pretrain-updates 10000 \
  --demo-batch-fraction 0.25 --bc-weight 1.0 --bc-decay-steps 250000
```

The expert data is collected only from the training reset distribution and is
kept in a separate, permanent replay buffer. Each online update contains 75%
online transitions and 25% demonstrations; an additional behavior-cloning
loss starts at weight 1.0 and linearly decays to zero over 250,000 environment
steps. TensorBoard logs `demo/*`, `pretrain/bc_loss`, `loss/bc`, and
`schedule/bc_weight`.

To retain the original pure-SAC baseline, disable demonstrations explicitly:

```bash
moebius-train --algorithm sac --total-steps 1000000 --demo-episodes 0
```

## Train PEARL with hidden mass and friction

Object and goal state is observable, but mass and sliding friction are not.
The context encoder must infer useful dynamics information from `(s, a, r,
s')` transitions. The expert demonstrations are restricted to training tasks;
held-out dynamics are never added to a buffer.

```bash
moebius-train --algorithm pearl --pearl-task-distribution dynamics \
  --num-train-tasks 20 --num-eval-tasks 10 \
  --latent-dim 5 --context-size 32 --meta-batch-size 8 \
  --mass-scale-min 0.5 --mass-scale-max 1.5 \
  --friction-scale-min 0.5 --friction-scale-max 1.5 \
  --pearl-demo-episodes-per-task 2 --pearl-bc-pretrain-updates 5000 \
  --demo-batch-fraction 0.25 --bc-weight 1.0 --bc-decay-steps 250000 \
  --pearl-bc-weight-floor 0.1 --context-episodes-per-task 64 \
  --total-steps 1000000
```

Set `--pearl-demo-episodes-per-task 0` to run the original random-exploration
PEARL variant. TensorBoard logs `demo/pearl_*`, `pretrain/pearl_bc_loss`,
`loss/bc`, and `schedule/pearl_bc_weight` alongside adaptation success.

The context buffers preserve episode order. During online fine-tuning, expert
actions are cloned under the learner's own current task context, not a
separate clean expert context. This aligns the latent used during training
with the empty/partial context available at deployment.

## Run the legacy Phase 2 goal task distribution

The prior fixed-goal PEARL experiment remains available for comparison:

```bash
moebius-train --algorithm pearl --pearl-task-distribution goal \
  --pearl-demo-episodes-per-task 0 --total-steps 250000
```

For a quick wiring check, shrink both training and meta-batch parameters so
each task receives data early:

```bash
moebius-train --algorithm pearl --pearl-task-distribution dynamics \
  --total-steps 5000 --start-steps 1000 \
  --num-train-tasks 4 --num-eval-tasks 2 --meta-batch-size 2 \
  --batch-size 32 --context-size 16 --pearl-demo-episodes-per-task 1 \
  --pearl-bc-pretrain-updates 100 --eval-interval 2500
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
moebius-record --checkpoint checkpoints/pick-place-v3_sac_seed1.pt --episodes 3
```

Record PEARL's adaptation on one held-out task instead:

```bash
moebius-record --checkpoint checkpoints/pick-place-v3_pearl_seed1.pt \
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
moebius-record --checkpoint checkpoints/pick-place-v3_sac_seed1.pt --camera-name corner2
```

The final checkpoint is saved to `checkpoints/`. Tests that do not require a
MuJoCo installation can be run with:

```bash
pytest
```
