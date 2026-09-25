# Moebius

State-based reinforcement-learning experiments for robot manipulation. Phase 1
is a Soft Actor-Critic (SAC) baseline, Phase 2 adds PEARL latent-task
inference, Phase 3 applies that pipeline to MetaWorld pick-and-place, and
Phase 4 introduces hidden object properties.


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

- Each PEARL task fixes an unobserved cylinder radius, height, mass,
  object/table friction, object/finger friction, and contact softness. The
  default bounds are deliberately broad and stored in `TrainConfig`, with
  independently sampled and disjoint train/test property vectors.
- Before training, each candidate property vector must yield
  `pearl_demo_episodes_per_task` successful scripted rollouts within
  `pearl_demo_max_attempts_per_task` attempts. Candidates that fail are
  resampled; successful training rollouts are retained directly as the PEARL
  demonstration buffer. The exact accepted train/test vectors are saved in the
  checkpoint.
- Object and goal positions are still randomized on every reset within each
  task. This ensures that PEARL must infer dynamics from transitions rather
  than using a visible configuration as a task label.
- PEARL behavior-clones both its actor and context encoder on the same
  context shapes used at deployment (see "Encoder context" below), including
  the empty context (`z = 0`) of each task's first episode. It then uses the retained task-specific demonstrations
  in mixed online/offline updates with a cloning-loss floor instead of removing
  the expert anchor entirely.
- PEARL now behavior-clones both the deterministic action and a conservative
  policy spread, then warms up its critics on demonstrations before online
  interaction. Online updates use scaled rewards, Huber critic losses,
  gradient clipping, and a Q-normalized actor objective so the cloning term
  cannot be overwhelmed by the critic scale. The normalization divides the
  whole SAC objective (entropy and Q) by the batch Q magnitude, so the
  entropy temperature keeps its usual meaning. The temperature is tuned
  automatically toward the target entropy, starting from
  `--pearl-initial-alpha` (0.02 in scaled-reward units); pass
  `--pearl-fixed-alpha` to hold it constant instead. TensorBoard logs
  `stats/policy_entropy` and `stats/policy_log_std`; an entropy near
  `action_dim * log(2)` (about 2.77 for MetaWorld) means the policy is close
  to uniform random actions.
- One task latent is sampled and held fixed for each complete rollout. Tasks
  are collected in short adaptation sequences, matching evaluation instead
  of resampling a different latent at every environment step. A posterior
  standard-deviation floor prevents the context encoder from becoming
  arbitrarily overconfident.

### Encoder context

Training, evaluation and recording all choose an episode's latent with one
rule (`PEARLAgent.latent_for_episode` and `moebius.context.ContextBuilder`):

- The first episode on a task uses `z = 0`.
- Before adaptation episode `k`, the context is pooled from all `k` completed
  earlier episodes on that task, not just the tail of the last one.
- Up to `--context-motion-fraction` (default 0.75) of the `--context-size`
  (default 64) transitions are ones where the object moved more than
  `--context-motion-threshold` (default 1 mm) in a step. Hidden mass,
  friction and softness only show up while the object is being handled.
  Motion is read from the observation, so nothing hidden is used.
- Training samples `k` from 0 to `--adaptation-episodes` and builds each
  task's context from the first `k` episodes of one of its
  `--context-sequences-per-task` (default 4) most recent adaptation sequences,
  so the encoder is trained on what it will see at deployment.

The encoder turns each transition into a Gaussian factor whose variance uses
`softplus` instead of a clamped log-variance. The clamp had zero gradient once
saturated, which left earlier runs with a posterior equal to the prior. The KL
penalty uses `--kl-weight` (default 0.05) with `--kl-free-bits` (default 0.1
nats per latent dimension that are not penalized), so the posterior is not
pushed all the way back to the prior.

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

## Train PEARL with hidden object properties

Each task samples a hidden cylinder radius, height, mass, object/table
friction, object/finger friction, and contact softness. The intentionally wide
bounds are stored in `TrainConfig`, rather than exposed as training-command
flags, so they remain part of each experiment's saved configuration. Object
and goal positions are randomized per reset; PEARL must infer task properties
from `(s, a, r, s')` transitions. Expert demonstrations are restricted to
training tasks; held-out dynamics are never added to a buffer.

```bash
moebius-train --algorithm pearl --pearl-task-distribution dynamics \
  --num-train-tasks 20 --num-eval-tasks 10 \
  --latent-dim 5 --context-size 64 --meta-batch-size 8 \
  --pearl-demo-episodes-per-task 5 --pearl-bc-pretrain-updates 5000 \
  --pearl-critic-pretrain-updates 5000 \
  --demo-batch-fraction 0.25 --bc-weight 1.0 --bc-decay-steps 500000 \
  --pearl-bc-weight-floor 0.2 --context-sequences-per-task 4 \
  --pearl-reward-scale 0.1 --pearl-initial-alpha 0.02 \
  --total-steps 1000000
```

Set `--pearl-demo-episodes-per-task 0` to run the original random-exploration
PEARL variant. TensorBoard logs `demo/pearl_*`, `pretrain/pearl_bc_loss`,
`pretrain/pearl_critic_loss`, `pretrain/pearl_joint_bc_loss`, gradient norms,
target/Q scales, `loss/bc`, and `schedule/pearl_bc_weight` alongside adaptation
success.

During online fine-tuning, expert
actions are cloned under the learner's own sampled task latent, not a separate
clean expert context. That latent remains fixed during a rollout and is
updated between adaptation episodes, matching deployment.

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

To check that task inference is alive:

- `stats/z_std` should fall clearly below 1, and `eval/z_std_after_context_k`
  should shrink as `k` grows.
- `stats/z_mean_task_std` should be well above 0. Near 0 means every task gets
  the same latent.
- `loss/kl` should settle above the free-bits floor (0.5 nats with 5 latent
  dimensions).
- `stats/context_motion_fraction` and `eval/context_motion_fraction_k` show
  how much of the context involves the object moving. Values near 0 mean the
  policy rarely touches the object, so there is little to infer from.

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
