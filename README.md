# Moebius

State-based reinforcement-learning experiments for robot manipulation. Phase 1
is a Soft Actor-Critic (SAC) baseline, Phase 2 adds PEARL latent-task
inference, Phase 3 applies that pipeline to MetaWorld pick-and-place, and
Phase 4 introduces object shapes and hidden object properties.


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

## Included in Phase 4 — Object shapes and hidden properties

- Each PEARL task picks an object shape (cylinder, box, bar or sphere,
  assigned round-robin so every shape is in both the train and test sets), a
  size, and three hidden properties: mass, finger friction and table
  friction. Mass together with finger friction decides whether a grasp slips;
  low table friction makes the object slide or roll away when touched. All
  shapes rest with the same orientation, so the observed object orientation
  does not reveal the shape. Bounds live in `TrainConfig`.
- The shape is swapped inside MetaWorld's pick-place model (MuJoCo primitive
  geoms can change type and size in place), so there is one environment and
  no modified copies of MetaWorld's scene files.
- MetaWorld's finger actuators are position servos with no force limit, so
  they squeeze far harder than any of these objects weigh and mass and
  friction barely affect the grasp. `--gripper-force-limit` caps each finger's
  force (a robot constant, not a task property). Use `moebius-sweep` to choose
  a value.
- Before training, each candidate object must yield
  `pearl_demo_episodes_per_task` successful scripted rollouts within
  `pearl_demo_max_attempts_per_task` attempts, from any of the expert variants
  in `--pearl-expert-variants` (tried in order; `metaworld` is MetaWorld's own
  scripted policy). Candidates that no variant solves are resampled with the
  same shape. The exact accepted tasks and the variant that solved each are
  saved in the checkpoint and the `tasks/*` TensorBoard text entries.
- Object and goal positions are still randomized on every reset within each
  task. This ensures that PEARL must infer dynamics from transitions rather
  than using a visible configuration as a task label.
- PEARL first behavior-clones only its actor, task-agnostically under
  `z = 0`. As in the original PEARL, the context encoder is trained only by
  the critic loss and KL term, in every phase; cloning through the encoder
  made the latent's meaning shift once critic training took over. It then
  uses the retained task-specific demonstrations
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

### Failed expert attempts

The scripted expert ignores the hidden properties, and only its successful
rollouts become demonstrations, so the demonstrations make every object look
equally easy. Its failed attempts on each accepted training task (the ones
made before it reached the required successes) are now added to the online
replay and context buffers. They show the same behavior leading to different
outcomes on different objects, which is the task-dependent value signal the
encoder needs. They are never behavior-cloned. Disable with
`--no-pearl-replay-failed-demos`; `demo/pearl_failed_episodes` counts them.

### Critic stability

The critic's TD targets are capped at the largest achievable soft return,
`(pearl_max_reward * pearl_reward_scale + alpha * max_entropy) / (1 - gamma)`,
where `max_entropy` is the entropy of a uniform policy over the action box.
With MetaWorld's maximum reward of 10 and a reward scale of 0.1 that is about
100 plus the entropy allowance. The critics also use LayerNorm, which limits
how far Q can extrapolate on unfamiliar actions and latents. An earlier run
without these safeguards reached Q values about 20 times the achievable
maximum. Use `--pearl-max-reward 0` or `--no-pearl-critic-layer-norm` to turn
them off. TensorBoard logs `stats/q_max`, `stats/q_upper_bound` and
`stats/target_q_clipped_fraction`. A clipped fraction that stays above a few
percent means the critic is still pushing against the cap.

### Encoder context

Training, evaluation and recording all choose an episode's latent with one
rule (`PEARLAgent.latent_for_episode` and `moebius.context.ContextBuilder`):

- The first episode on a task uses `z = 0`.
- Before adaptation episode `k`, the context is pooled from all `k` completed
  earlier episodes on that task, not just the tail of the last one.
- Up to `--context-motion-fraction` (default 0.75) of the `--context-size`
  (default 64) transitions are ones where the object moved more than
  `--context-motion-threshold` (default 1 mm) in a step. Hidden mass
  and friction only show up while the object is being handled.
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

## Check that objects call for different behavior

Before training, run the sweep. For each gripper force limit, it tries every
expert variant on every shape at low/middle/high mass and finger friction and
low/high table friction, then reports the *adaptation gap*: success when the
best variant is picked for each object, minus success of the best single
variant. A gap near zero means one fixed behavior handles every object and
PEARL's latent has nothing to add. Read the *held-out* gap: variants are
chosen on half of the episodes and scored on the other half. The in-sample
gap picks the best of several noisy estimates and is positive even when no
real difference exists. Use `--episodes 4` or more. The sweep also prints the
measured grip force.

```bash
pip install -e .   # registers the moebius-sweep command
moebius-sweep --force-limits none 40 --episodes 4           # quick first pass
moebius-sweep --force-limits 60 --masses 0.1 0.6 1.2 --max-steps 500 200 150 --episodes 4   # time pressure
moebius-sweep --episodes 5 --output runs/sweep_full.csv     # full grid
```

`--max-steps` evaluates several episode lengths from the same episodes
(each runs once at the longest length; success is read at every shorter one).
With 500 steps, slow and careful handling costs nothing, so one careful
behavior tends to win everywhere; shorter episodes make it costly on easy
objects, which is what can make the best behavior differ between objects.

The full default grid is 4 force limits x 72 objects x 4 variants x 3
episodes, about an hour on a laptop. Pick the force limit with the largest
gap at which most objects are still solvable, and pass it to training as
`--gripper-force-limit`.

## Train PEARL with object shapes and hidden properties

Each task samples a shape, a size and hidden mass, finger friction and table
friction. The bounds are stored in `TrainConfig`, rather than exposed as
training-command flags, so they remain part of each experiment's saved
configuration. Object
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
  --gripper-force-limit 60 --max-episode-steps 150 \
  --pearl-eval-sequences-per-task 3 --eval-interval 30000 \
  --total-steps 1000000
```

The 60 N force limit and 150-step episodes come from the sweep: under them,
the careful expert is best for boxes and bars, while slow transport with
MetaWorld's higher grasp is best for cylinders (54% vs 15%), and neither shape
can be told apart before contact. Defaults leave out spheres and cap mass at
1.2 kg, because the sweep found those objects mostly unsolvable by every
expert variant.

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

Evaluation follows the original PEARL protocol. Every held-out episode samples
`z` from the current posterior (`z = 0` before any context) and uses the
policy's mean action. Adaptation step `k` is therefore the only thing that
changes between `eval/success_after_context_0` and
`eval/success_after_context_k`, and `eval/adaptation_gain` (last step minus
the first) measures what context adds. Each evaluation runs
`--pearl-eval-sequences-per-task` independent adaptation sequences per
held-out task (default 1; `eval/episodes_per_step` reports the resulting
sample size). With 10 held-out tasks, one sequence gives only 10 episodes per
step, roughly +/-14 points of noise at 30% success. For comparisons between
runs, use 3 or more sequences with a longer `--eval-interval` (for example
`--pearl-eval-sequences-per-task 3 --eval-interval 30000`), or average
several evaluations.

Per-task results are logged as `eval_task/task_<id>_success` and
`train_task/task_<id>_success` / `_return`. The `tasks/train` and `tasks/test`
text entries list each task's hidden property values.

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

This produces one context-collection video per adaptation episode, followed by
an `*_adapted.mp4` rollout. Every episode follows the evaluation protocol: `z`
is sampled from the posterior over the earlier episodes, and actions are the
policy's mean. Omit `--adaptation-episodes` to use the value saved in the checkpoint.
Use `--task-index` to select another held-out task configuration.

Videos are written to `videos/`. Use `--video-dir` or `--fps` to change the
output folder or frame rate. Recording uses the `corner` camera by default.
MuJoCo renders that camera upside down, so its frames are rotated 180 degrees
automatically. Other camera angles are available, for example the top-down
view:

```bash
moebius-record --checkpoint checkpoints/pick-place-v3_sac_seed1.pt --camera-name topview
```

To correct another camera that renders upside down, add its name to
`UPSIDE_DOWN_CAMERAS` in `record.py`.

The final checkpoint is saved to `checkpoints/`. Tests that do not require a
MuJoCo installation can be run with:

```bash
pytest
```
